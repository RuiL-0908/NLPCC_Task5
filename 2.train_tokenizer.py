import json
import os
import shutil
from datasets import load_dataset
from transformers import AutoTokenizer

# --- 2.1 训练基础 Tokenizer ---
print("加载基础分词器 (继承 [CLS], [SEP], [MASK] 规则)...")
old_tokenizer = AutoTokenizer.from_pretrained("IDEA-CCNL/Erlangshen-DeBERTa-v2-97M-Chinese")

print("加载官方 100M 训练数据用于构建词表...")
raw_dataset = load_dataset("chinese-babylm-org/babylm-zho-100M")
dataset = raw_dataset["train"] if "train" in raw_dataset else raw_dataset

def get_training_corpus():
    for i in range(0, len(dataset), 1000):
        yield dataset[i : i + 1000]["text"]

print("正在训练全新 Tokenizer...")
new_tokenizer = old_tokenizer.train_new_from_iterator(get_training_corpus(), vocab_size=12800)
OLD_PATH = "./babylm-chinese-deberta-v2-14M-tokenizer"
new_tokenizer.save_pretrained(OLD_PATH)

# --- 2.2 替换手术: 强行注入中文全角逗号 ---
NEW_PATH = "./babylm-chinese-deberta-v2-14M-tokenizer-replaced"
if os.path.exists(NEW_PATH):
    shutil.rmtree(NEW_PATH)
shutil.copytree(OLD_PATH, NEW_PATH)

tokenizer_file = os.path.join(NEW_PATH, "tokenizer.json")
with open(tokenizer_file, "r", encoding="utf-8") as f:
    tokenizer_data = json.load(f)

vocab = tokenizer_data["model"]["vocab"]
inv_vocab = {v: k for k, v in vocab.items()}

target_char = "，"
if target_char not in vocab:
    target_vocab_size = len(vocab)
    victim_id = target_vocab_size - 1
    victim_word = inv_vocab[victim_id]
    
    del vocab[victim_word]
    vocab[target_char] = victim_id
    
    tokenizer_data["model"]["vocab"] = vocab
    with open(tokenizer_file, "w", encoding="utf-8") as f:
        json.dump(tokenizer_data, f, ensure_ascii=False, indent=2)
    print(f"被替换: [{victim_word}] (ID:{victim_id}) -> 成功注入: [{target_char}]")
else:
    print(f"词表已包含 '{target_char}'，无需替换。")