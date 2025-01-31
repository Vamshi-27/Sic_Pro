import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer

# Load dataset
dataset = load_from_disk("./data/writing_prompts")

# Use a public model as fallback
model_name = "EleutherAI/gpt-neo-2.7B"  # Fallback to a public model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

# Tokenization function
def tokenize_function(examples):
    return tokenizer(examples["story"], padding="max_length", truncation=True, max_length=512)

# Tokenize dataset
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Training arguments
training_args = TrainingArguments(
    output_dir="./models/deepseek_v3_finetuned",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=2,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
)

# Train model
trainer.train()

# Save model and tokenizer
model.save_pretrained("./models/deepseek_v3_finetuned")
tokenizer.save_pretrained("./models/deepseek_v3_finetuned")
