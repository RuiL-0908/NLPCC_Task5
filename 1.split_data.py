import os
from datasets import load_dataset
from transformers import AutoTokenizer

print("1. 加载 Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("IDEA-CCNL/Erlangshen-DeBERTa-v2-97M-Chinese")

print("2. 从 Hugging Face 加载官方 100M 数据集...")
raw_dataset = load_dataset("chinese-babylm-org/babylm-zho-100M")
dataset = raw_dataset["train"] if "train" in raw_dataset else raw_dataset

def compute_length(example):
    tokens = tokenizer(example["text"], truncation=False)
    return {"length": len(tokens["input_ids"])}

print("3. 正在精确计算每句文本的 Token 长度 (多进程加速中)...")
dataset_with_length = dataset.map(compute_length, num_proc=8)
dataset_valid = dataset_with_length.filter(lambda x: x["length"] > 2, num_proc=8)

print("\n4. 开始物理切分数据集并落盘...")
# 阶段一 (Easy) -> <= 32
easy_dataset = dataset_valid.filter(lambda x: x["length"] <= 32, num_proc=8)
easy_dataset.save_to_disk("./data_easy")

# 阶段二 (Medium) -> <= 64
medium_dataset = dataset_valid.filter(lambda x: x["length"] <= 64, num_proc=8)
medium_dataset.save_to_disk("./data_medium")

# 阶段三 (Hard) -> <= 128
hard_dataset = dataset_valid.filter(lambda x: x["length"] <= 128, num_proc=8)
hard_dataset.save_to_disk("./data_hard")

# 阶段四 (Full) -> 全量
dataset_valid.save_to_disk("./data_full")
print("物理切分完成，四个难度的数据集已保存在当前目录。")