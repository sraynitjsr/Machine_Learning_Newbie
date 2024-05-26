from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load pre-trained model and tokenizer
model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Define pad token id
tokenizer.pad_token_id = tokenizer.eos_token_id

# List of prompts
prompts = [
    "Sachin Tendulkar",
    "The future of artificial intelligence",
    "Once upon a time in a faraway land",
    "The impact of climate change on global ecosystems",
    "A day in the life of a software developer"
]

for prompt in prompts:
    # Encode the input prompt
    input_ids = tokenizer.encode(prompt, return_tensors='pt')

    # Create attention mask
    attention_mask = (input_ids != tokenizer.pad_token_id).long()

    # Generate text
    output = model.generate(input_ids, attention_mask=attention_mask, max_length=50, num_return_sequences=1)

    # Decode the generated text
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

    # Print the generated text
    print(f"Prompt: {prompt}")
    print(generated_text)
    print("\n" + "-"*50 + "\n")
