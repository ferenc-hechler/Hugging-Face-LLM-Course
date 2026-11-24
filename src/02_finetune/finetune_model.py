# see https://huggingface.co/docs/transformers/training for more details
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorForLanguageModeling
import time
import torch; 


print(f"GPU: {torch.cuda.is_available()}")

# model_name = "Qwen/Qwen3-0.6B"
model_name = "qwen3-finetuned/v1"
new_model_name = "qwen3-finetuned"
# model_name = "google/gemma-3-270m-it"
# new_model_name = "gemma3-finetuned"


start = time.time()


tokenizer = AutoTokenizer.from_pretrained(model_name)
dataset = load_dataset("karthiksagarn/astro_horoscope", split="train")

def tokenize(batch):
    return tokenizer(
        batch["horoscope"],
        truncation=True,
        max_length=512,
    )

print("tokenizing...")
dataset = dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)

print("splitting...")
dataset = dataset.train_test_split(test_size=0.1)

data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

from transformers import AutoModelForCausalLM, TrainingArguments, Trainer

model = AutoModelForCausalLM.from_pretrained(model_name, dtype="auto")

training_args = TrainingArguments(
    output_dir=new_model_name,
    num_train_epochs=7,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1, 
    gradient_accumulation_steps=16,
    gradient_checkpointing=True,
    # gradient_checkpointing_kwargs={"every_n_layers": 4},
    bf16=True,
    learning_rate=2e-5,
    logging_steps=10,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    prediction_loss_only=True,             # Logits nicht sammeln
)

prep = time.time()
print(f"Preprocessing time: {(prep - start)/60} minutes")


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    processing_class=tokenizer,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

print(f"training {new_model_name} ...")
trainer.train()
# trainer.push_to_hub()

print("saving...")
trainer.save_model(f"{new_model_name}/final")
tokenizer.save_pretrained(f"{new_model_name}/final")

stop = time.time()
print(f"Training time: {(stop - start)/60} minutes")

print("finished")