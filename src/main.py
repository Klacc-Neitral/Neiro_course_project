import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(
    "../models/checkpoints/checkpoint-2200"
)
model = AutoModelForSeq2SeqLM.from_pretrained(
    "../models/checkpoints/checkpoint-2200"
).to(device)

text = "안녕"
inputs = tokenizer(text, return_tensors="pt").to(device)

out = model.generate(
    **inputs,
    max_new_tokens=30,
    do_sample=True,
    top_p=0.9,
    temperature=0.8
)

print(tokenizer.decode(out[0], skip_special_tokens=True))
