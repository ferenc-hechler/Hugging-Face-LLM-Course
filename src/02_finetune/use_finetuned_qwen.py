from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

path = "qwen3-finetuned/final"
tokenizer = AutoTokenizer.from_pretrained(path)
model = AutoModelForCausalLM.from_pretrained(path, dtype="auto")

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
print(pipe("Today your horoscope says", max_new_tokens=100)[0]["generated_text"])
