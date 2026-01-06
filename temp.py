import random
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")

# Get vocab as token -> id mapping
vocab = tokenizer.get_vocab()
token_ids = list(vocab.values())

# Sample 50 random token IDs
sample_ids = random.sample(token_ids, 50)

# Decode and print them
for tid in sample_ids:
    decoded = tokenizer.decode([tid], clean_up_tokenization_spaces=False)
    print(f"{tid:>6} -> {repr(decoded)}")
