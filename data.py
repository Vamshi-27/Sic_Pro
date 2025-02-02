from datasets import load_dataset

dataset = load_dataset("euclaise/writingprompts")
dataset.save_to_disk("./data/writing_prompts")
