
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import get_peft_model, LoraConfig, TaskType
from datasets import load_dataset
import torch
import os

# ✅ Load model + tokenizer from GCS path
model_name_or_path = "gs://finn_training/llama3_2b_model/Llama-3.2-3B/"

tokenizer = AutoTokenizer.from_pretrained(
    model_name_or_path, trust_remote_code=True
)

model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    torch_dtype=torch.float16,
    trust_remote_code=True
)

# ✅ Inject LoRA adapter
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "v_proj"]
)
model = get_peft_model(model, peft_config)

# ✅ Tokenization function
def tokenize_fn(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=512)

collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# === PHASE 1 ===
print("🔁 Loading Phase 1 Dataset (Elon + Snailbrook)...")
dataset_phase1 = load_dataset("json", data_files={"train": "gs://finn_training/send_bucket/phase1.jsonl"}, streaming=False)["train"]
tokenized_phase1 = dataset_phase1.map(tokenize_fn, batched=True)

print("🚀 Starting Phase 1 Fine-Tuning...")
args1 = TrainingArguments(
    output_dir="./finn_phase1_model",
    logging_dir="./logs_phase1",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    learning_rate=2e-5,
    lr_scheduler_type="cosine",
    logging_steps=50,
    save_steps=500,
    save_total_limit=2,
    fp16=True,
    seed=42
)

trainer1 = Trainer(
    model=model,
    args=args1,
    train_dataset=tokenized_phase1,
    tokenizer=tokenizer,
    data_collator=collator,
)
trainer1.train()

# ✅ Save Phase 1 Model
print("💾 Saving Phase 1 Output...")
model.save_pretrained("./finn_phase1_model")
tokenizer.save_pretrained("./finn_phase1_model")
os.system("gsutil cp -r ./finn_phase1_model gs://finn_training/send_bucket/finn_phase1/model/")

# === PHASE 2 ===
print("🔁 Loading Phase 2 Dataset (4chan + Simulation)...")
dataset_phase2 = load_dataset("json", data_files={"train": "gs://finn_training/send_bucket/phase2.jsonl"}, streaming=False)["train"]
tokenized_phase2 = dataset_phase2.map(tokenize_fn, batched=True)

print("🚀 Starting Phase 2 Fine-Tuning...")
args2 = TrainingArguments(
    output_dir="./finn_phase2_model",
    logging_dir="./logs_phase2",
    num_train_epochs=2,
    per_device_train_batch_size=2,
    learning_rate=5e-6,
    lr_scheduler_type="cosine",
    logging_steps=50,
    save_steps=500,
    save_total_limit=2,
    fp16=True,
    seed=42
)

trainer2 = Trainer(
    model=model,
    args=args2,
    train_dataset=tokenized_phase2,
    tokenizer=tokenizer,
    data_collator=collator,
)
trainer2.train()

# ✅ Save Final Model
print("💾 Saving Final Governor Finn Model to GCS...")
model.save_pretrained("./final_finn_model")
tokenizer.save_pretrained("./final_finn_model")
os.system("gsutil cp -r ./final_finn_model gs://finn_training/send_bucket/final_finn_model/")
