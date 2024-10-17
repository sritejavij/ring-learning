import torch
import gc
import os
import psutil
from transformers import LlamaForCausalLM, LlamaTokenizer
from easy_context import (
    prepare_seq_parallel_inputs,
    apply_seq_parallel_monkey_patch,
    prepare_dataloader,
)
from accelerate import Accelerator
from torch.utils.data import DataLoader, Dataset
import torch.profiler

# Function to apply the desired attention mechanism
def apply_attention_patch(attention_type, model_type="llama"):
    """
    Applies the appropriate attention mechanism based on the attention type.

    Args:
        attention_type (str): Type of attention ('regular' or 'ring').
        model_type (str): Model type (default is 'llama').
    """
    if attention_type == "ring":
        apply_seq_parallel_monkey_patch("zigzag_ring_attn", model_type)
    elif attention_type == "regular":
        # No patch needed for regular attention as it's the default
        pass
    else:
        raise ValueError(f"Unknown attention type: {attention_type}")

# Function to initialize model and tokenizer
def initialize_model(model_name, attention_type):
    """
    Initializes the tokenizer and model with the specified attention mechanism.

    Args:
        model_name (str): The pre-trained model name.
        attention_type (str): Type of attention mechanism ('regular' or 'ring').

    Returns:
        model (LlamaForCausalLM): The initialized model.
        tokenizer (LlamaTokenizer): The initialized tokenizer.
    """
    tokenizer = LlamaTokenizer.from_pretrained(model_name)
    
    # Apply attention patch if necessary
    apply_attention_patch(attention_type, "llama")
    
    # Configure attention implementation
    if attention_type == "ring":
        attn_impl = "flash_attention_2"  # Required for ring attention
    else:
        attn_impl = None  # Standard attention without Flash Attention

    # Load the model with the specified attention implementation
    model = LlamaForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16,
        _attn_implementation=attn_impl,  # Flash Attention 2 for ring, None for regular
    )
    
    return model, tokenizer

# Synthetic Dataset
class RandomTextDataset(Dataset):
    """
    Creates a synthetic dataset with random token IDs for testing.

    Args:
        tokenizer (LlamaTokenizer): The tokenizer to determine vocabulary size.
        length (int): The length of each sequence.
        num_samples (int): Number of samples in the dataset.
    """
    def __init__(self, tokenizer, length, num_samples=1):
        self.tokenizer = tokenizer
        self.length = length
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        input_ids = torch.randint(
            low=0,
            high=len(self.tokenizer),
            size=(self.length,),
            dtype=torch.long,
        )
        position_ids = torch.arange(self.length, dtype=torch.long)
        target_ids = input_ids.clone()
        return {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "target_ids": target_ids,
        }

# Collate Function
def collate_fn(batch):
    """
    Collates a batch of samples into tensors.

    Args:
        batch (list): List of samples.

    Returns:
        dict: Dictionary containing batched input_ids, position_ids, and target_ids.
    """
    return {
        "input_ids": torch.stack([item["input_ids"] for item in batch]),
        "position_ids": torch.stack([item["position_ids"] for item in batch]),
        "target_ids": torch.stack([item["target_ids"] for item in batch]),
    }

# Function to get current memory usage per GPU
def get_per_gpu_memory_usage():
    """
    Retrieves the current memory usage for each GPU.

    Returns:
        dict: Dictionary mapping GPU indices to memory usage in GB.
    """
    if torch.cuda.is_available():
        memory_usage = {}
        for device in range(torch.cuda.device_count()):
            mem = torch.cuda.memory_allocated(device) / (1024 ** 3)  # GB
            memory_usage[f"GPU {device}"] = mem
        return memory_usage
    else:
        process = psutil.Process(os.getpid())
        return {"CPU Memory": process.memory_info().rss / (1024 ** 3)}  # GB

# Function to get peak memory usage per GPU
def get_per_gpu_peak_memory_usage():
    """
    Retrieves the peak memory usage for each GPU since the last reset.

    Returns:
        dict: Dictionary mapping GPU indices to peak memory usage in GB.
    """
    if torch.cuda.is_available():
        memory_usage = {}
        for device in range(torch.cuda.device_count()):
            mem = torch.cuda.max_memory_allocated(device) / (1024 ** 3)  # GB
            memory_usage[f"GPU {device}"] = mem
        return memory_usage
    else:
        process = psutil.Process(os.getpid())
        return {"CPU Memory": process.memory_info().rss / (1024 ** 3)}  # GB

# Function to perform the test
def test_attention(
    model_name, attention_type, context_lengths, batch_size=1, num_samples=1
):
    """
    Tests the memory usage of the specified attention mechanism across different context lengths.

    Args:
        model_name (str): The pre-trained model name.
        attention_type (str): Type of attention mechanism ('regular' or 'ring').
        context_lengths (list): List of context lengths to test.
        batch_size (int): Batch size for testing.
        num_samples (int): Number of samples per batch.
    """
    print(f"\n=== Testing {attention_type.capitalize()} Attention ===\n")
    
    # Initialize Accelerator
    accelerator = Accelerator()
    
    # Initialize model and tokenizer
    model, tokenizer = initialize_model(model_name, attention_type)
    
    # Confirm attention implementation
    print(f"Attention Implementation: {model.config._attn_implementation}")
    if attention_type == "ring":
        print("Ring Attention is enabled.")
    else:
        print("Regular Attention is enabled.")
    
    # Context lengths to test
    for length in context_lengths:
        print(f"\nTesting context length: {length}")
        
        # Prepare dataset and dataloader
        dataset = RandomTextDataset(tokenizer, length, num_samples=batch_size)
        dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)
        
        # Apply EasyContext modifications if using ring attention
        if attention_type == "ring":
            prepare_dataloader("zigzag_ring_attn", dataloader, accelerator)
        
        # Prepare model and dataloader with Accelerator
        model, dataloader = accelerator.prepare(model, dataloader)
        
        # Clear cache and reset peak memory stats
        if torch.cuda.is_available():
            for device in range(torch.cuda.device_count()):
                torch.cuda.reset_peak_memory_stats(device)
                torch.cuda.empty_cache()
        gc.collect()
        
        # Measure peak memory before
        mem_before = get_per_gpu_peak_memory_usage()
        
        # Run the model with profiling
        model.eval()
        with torch.no_grad():
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                record_shapes=True,
                profile_memory=True,
                with_stack=True,
            ) as prof:
                for batch in dataloader:
                    if attention_type == "ring":
                        # Shard the sequences for ring attention
                        prepared = prepare_seq_parallel_inputs(
                            "zigzag_ring_attn",
                            batch["input_ids"],
                            batch["position_ids"],
                            batch["target_ids"],
                            accelerator.process_index,
                            accelerator.num_processes,
                            accelerator.device,
                        )
                        local_input_ids = prepared["local_input_ids"].detach()
                        local_position_ids = prepared["local_position_ids"].detach()
                        local_target_ids = prepared["local_target_ids"].detach()
                        
                        # Forward pass
                        outputs = model(local_input_ids, position_ids=local_position_ids)
                    else:
                        # Regular attention forward pass
                        outputs = model(
                            batch["input_ids"].to(accelerator.device),
                            position_ids=batch["position_ids"].to(accelerator.device),
                        )
                    logits = outputs.logits
                    
                    # Break after one batch to measure memory usage
                    break
        
        # Measure peak memory after
        mem_after = get_per_gpu_peak_memory_usage()
        
        # Calculate memory used per GPU
        mem_used = {}
        for key in mem_after:
            mem_used[key] = mem_after[key] - mem_before.get(key, 0)
        
        # Print memory usage
        for gpu, mem in mem_used.items():
            print(f"{gpu} Memory Used: {mem:.3f} GB")
        print("\n" + "-" * 50 + "\n")
        
        # Optionally, print profiler results
        # print(prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=10))

# Main Execution
if __name__ == "__main__":
    # Configuration
    MODEL_NAME = "huggyllama/llama-7b"  # Adjust as needed
    CONTEXT_LENGTHS = [512, 1024, 2048, 4096, 8192]  # Context lengths to test
    BATCH_SIZE = 1  # Fixed batch size
    NUM_SAMPLES = 1  # Number of samples per batch

    # Test Regular Attention
    test_attention(
        model_name=MODEL_NAME,
        attention_type="regular",
        context_lengths=CONTEXT_LENGTHS,
        batch_size=BATCH_SIZE,
        num_samples=NUM_SAMPLES,
    )
    
    # Test Ring Attention
    test_attention(
        model_name=MODEL_NAME,
        attention_type="ring",
        context_lengths=CONTEXT_LENGTHS,
        batch_size=BATCH_SIZE,
        num_samples=NUM_SAMPLES,
    )
