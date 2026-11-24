from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# path = "Qwen/Qwen3-0.6B"
path = "qwen3-finetuned/v2"
# path = "google/gemma-3-270m-it"
# path = "gemma3-finetuned/v1"

tokenizer = AutoTokenizer.from_pretrained(path)
model = AutoModelForCausalLM.from_pretrained(path, dtype="auto")

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

for i in range(10):
    print(f"\n{path} {i}: " + pipe("Today your horoscope says", max_new_tokens=100)[0]["generated_text"])
