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

# Swap attention implementation to 'zigzag_ring_attn' (Ring Attention *double check this is the right lib*)
apply_seq_parallel_monkey_patch("zigzag_ring_attn", "llama")

# Initialize the model and tokenizer
model_name = "huggyllama/llama-7b"  # Adjust as needed
tokenizer = LlamaTokenizer.from_pretrained(model_name)
model = LlamaForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float16,
    _attn_implementation="flash_attention_2",  # Toggle on flash_attention_2
)

# Initialize the Accelerator
accelerator = Accelerator()

# Define a synthetic dataset
class RandomTextDataset(Dataset):
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

def collate_fn(batch):
    return {
        "input_ids": torch.stack([item["input_ids"] for item in batch]),
        "position_ids": torch.stack([item["position_ids"] for item in batch]),
        "target_ids": torch.stack([item["target_ids"] for item in batch]),
    }

def get_memory_usage():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        mem_allocated = torch.cuda.max_memory_allocated()
        return mem_allocated / (1024 ** 3)  # In GB
    else:
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 ** 3)  # In GB

# Context lengths to test
context_lengths = [512, 1024, 2048, 4096, 8192]

for length in context_lengths:
    print(f"Testing context length: {length}")
    # Prepare dataset and dataloader
    dataset = RandomTextDataset(tokenizer, length)
    dataloader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn)

    # Prepare dataloader with EasyContext
    prepare_dataloader("zigzag_ring_attn", dataloader, accelerator)

    # Prepare model and dataloader with Accelerator
    model, dataloader = accelerator.prepare(model, dataloader)

    # Clear cache
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
    gc.collect()

    # Measure memory before
    mem_before = get_memory_usage()

    # Run the model
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            # Shard the sequences
            prepared = prepare_seq_parallel_inputs(
                "zigzag_ring_attn",
                batch["input_ids"],
                batch["position_ids"],
                batch["target_ids"],
                accelerator.process_index,
                accelerator.num_processes,
                accelerator.device,
            )
            local_input_ids = prepared["local_input_ids"]
            local_position_ids = prepared["local_position_ids"]
            local_target_ids = prepared["local_target_ids"]

            # Forward pass
            outputs = model(local_input_ids, position_ids=local_position_ids)
            logits = outputs.logits

            # Break after one batch to measure memory usage
            break

    # Measure memory after
    mem_after = get_memory_usage()

    # Calculate memory used
    mem_used = mem_after - mem_before

    print(f"Context Length: {length}, Memory Used: {mem_used:.3f} GB\n")
