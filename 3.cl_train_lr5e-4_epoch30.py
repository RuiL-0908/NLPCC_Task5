import torch
from datasets import load_from_disk
from transformers import (
    AutoTokenizer, DebertaV2Config, DebertaV2ForMaskedLM,
    DataCollatorForLanguageModeling, TrainingArguments, Trainer,
    set_seed
)

GLOBAL_SEED = 42
set_seed(GLOBAL_SEED)
BASE_DIR = "."

# 加载我们在上一步生成的替换版 Tokenizer
tokenizer = AutoTokenizer.from_pretrained("./babylm-chinese-deberta-v2-14M-tokenizer-replaced")
config = DebertaV2Config.from_json_file("config.json")
model = DebertaV2ForMaskedLM(config)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

curriculum_plan = [
    {"name": "幼教期 (Easy)",   "data": f"{BASE_DIR}/data_easy",   "epochs": 2, "out": f"{BASE_DIR}/babylm-chinese-deberta-v2-stage1-easy-lr5e-4-epoch30"},
    {"name": "小学期 (Medium)", "data": f"{BASE_DIR}/data_medium", "epochs": 3, "out": f"{BASE_DIR}/babylm-chinese-deberta-v2-stage2-medium-lr5e-4-epoch30"},
    {"name": "中学期 (Hard)",   "data": f"{BASE_DIR}/data_hard",   "epochs": 5, "out": f"{BASE_DIR}/babylm-chinese-deberta-v2-stage3-hard-lr5e-4-epoch30"},
    {"name": "冲刺期 (Full)",   "data": f"{BASE_DIR}/data_full",   "epochs": 20, "out": f"{BASE_DIR}/babylm-chinese-deberta-v2-final-lr5e-4-epoch30"}
]

for stage in curriculum_plan:
    print(f"\n🚀 开始阶段训练: {stage['name']} | 目标 Epoch 数: {stage['epochs']}")
    def tokenize_func(examples):
        return tokenizer(examples["text"], truncation=True, max_length=512, return_special_tokens_mask=True)
        
    raw_dataset = load_from_disk(stage["data"])
    current_dataset = raw_dataset["train"] if "train" in raw_dataset else raw_dataset
    train_dataset = current_dataset.map(tokenize_func, batched=True, num_proc=4, remove_columns=["text", "length"])
    
    training_args = TrainingArguments(
        output_dir=stage["out"],
        num_train_epochs=stage["epochs"],
        per_device_train_batch_size=32,
        gradient_accumulation_steps=4,
        learning_rate=5e-4, 
        warmup_ratio=0.05,
        optim="adamw_torch",      
        weight_decay=0.01, 
        seed=GLOBAL_SEED,           
        data_seed=GLOBAL_SEED,      
        bf16=False,
        fp16=False,
        save_strategy="no", 
        logging_steps=100,
        report_to="none"
    )
    
    trainer = Trainer(
        model=model, 
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )
    
    trainer.train()
    trainer.save_model(stage["out"])
    tokenizer.save_pretrained(stage["out"])