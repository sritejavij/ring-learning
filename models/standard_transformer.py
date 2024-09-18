!pip install transformers datasets accelerate
from datasets import load_dataset
import torch
from transformers import GPT2LMHeadModel, AutoTokenizer

dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
train_dataset = dataset['train']

model_name = "gpt2"  # A smaller 124M parameter model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

tokenizer.pad_token = tokenizer.eos_token  # Set the pad token to the eos token

# Load the GPT-2 model and set pad_token_id in the model config
model = GPT2LMHeadModel.from_pretrained(model_name, pad_token_id=tokenizer.eos_token_id)

# Function to generate long text sequences
def generate_long_text(base_text, repetitions):
    return base_text * repetitions

# Start testing
base_text = "Once upon a time, "  # A simple base text that will be repeated
max_length = 0

for i in range(1, 150):  # Gradually increase repetitions; GPT-2 can handle up to 1024 tokens
    try:
        # Generate a long sequence by repeating the base text
        long_input = generate_long_text(base_text, i)
        inputs = tokenizer(long_input, return_tensors="pt", padding=True, truncation=True)
        
        # Pass the attention mask
        attention_mask = inputs['attention_mask']
        
        # Check the number of tokens in the input
        num_tokens = inputs['input_ids'].shape[1]
        print(f"Trying with {num_tokens} tokens...")

        # Test if the model can handle the input
        with torch.no_grad():
            outputs = model.generate(inputs['input_ids'], attention_mask=attention_mask, max_length=num_tokens + 10)

        # If successful, record the max length
        max_length = num_tokens
    except Exception as e:
        print(f"Model breaks at {num_tokens} tokens with error: {e}")
        break

print(f"Maximum working token length: {max_length} tokens")


