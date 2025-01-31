import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load fine-tuned model and tokenizer
model_path = "./models/deepseek_v3_finetuned"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")

def generate_story():
    prompt = "Once upon a time,"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")

    output = model.generate(input_ids, max_length=500, temperature=0.8, top_p=0.95, do_sample=True)

    story = tokenizer.decode(output[0], skip_special_tokens=True)
    return story

if __name__ == "__main__":
    print(generate_story())
