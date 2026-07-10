

#  中文 BabyLM - DeBERTa-V2 (14M) 
**挑战赛:** 2026 中文 BabyLM 挑战赛 (2026 Chinese BabyLM Challenge)

##  模型简介

本模型为 DeBERTa-v2 掩码语言模型（约 14.67M 参数，12层 / 256隐藏层维度 / 8注意力头）。模型在官方 102M 中文 BabyLM 语料上从头训练了 30 个 epoch，并采用了一种基于物理数据切分的 4 阶段课程学习（Curriculum Learning）策略。

本仓库包含了从原始数据处理到最终评测的完整、可复现的全流程代码。

##  仓库文件结构

* `README.md` - 模型说明与复现指南
* `requirements.txt` - 精确的环境依赖版本
* `split_data.py` - 课程学习数据解析与物理切分脚本
* `train_tokenizer.py` - 自定义词表构建与手动 Token 映射修复脚本
* `cl_train_lr5e-4_epoch30.py` - 核心多阶段训练脚本
* `config.json` - DeBERTa-v2 14M 模型架构配置
* `myconfig.yaml` - 官方流水线评测配置

##  模型架构与配置

本模型旨在极小的参数预算下，榨取最大的认知能力：
* **架构:** DeBERTa-v2 (Masked Language Modeling)
* **总参数量:** 14,674,688 (~14.67M)
* **层数 (Layers):** 12
* **隐藏层维度 (Hidden Size):** 256
* **注意力头数 (Attention Heads):** 8
* **最大序列长度 (Max Sequence Length):** 512

##  分词器 (Tokenizer)

为了优化嵌入层与编码器的参数比例，我们沿用了 `BertTokenizer` 架构（继承自二郎神 Erlangshen-DeBERTa-v2），并在 BabyLM 语料上从头训练了一个高度压缩的自定义子词词表：
* **分词器类:** `BertTokenizer`
* **词表大小:** 12,800 
* **注:** 我们对训练后的 tokenizer 进行了手动微创修改，将词表最后的 ID 替换为中文全角逗号（`，`），以确保中文标点被正确切词。

##  训练数据与策略

* **数据集:** 官方中文 BabyLM 语料库 (102M)，通过脚本自动拉取。
* **课程学习 (Curriculum Learning):** 我们精确计算了每条序列的 Token 长度，并将数据集物理切分为四个渐进难度：
  1. **幼教期 / Easy (长度 <= 32):** 训练 2 个 Epochs
  2. **小学期 / Medium (长度 <= 64):** 训练 3 个 Epochs
  3. **中学期 / Hard (长度 <= 128):** 训练 5 个 Epochs
  4. **冲刺期 / Full Contexts (最大长度 512):** 训练 20 个 Epochs
* **核心超参数:** 峰值学习率 `5e-4`, Batch Size `32` (梯度累加 `4`), 优化器 `AdamW`, 全精度 `FP32`。

##  评估结果

官方基准测试 (14.67M 模型):

| 任务 | 分数 | 任务 | 分数 |
| :--- | :---: | :--- | :---: |
| **ZhoBLiMP** | 75.23 | **C3** | 42.45 |
| **AFQMC** | 71.11 | **Hanzi Structure** | 61.00 |
| **OCNLI** | 68.00 | **Hanzi Structure (Hidden)**| 60.25 |
| **CLUEWSC2020** | 64.14 | **Hanzi Pinyin** | 51.40 |
| **XComps-zh** | 57.85 | **Hanzi Pinyin (Hidden)**| 50.20 |
| **TNEWS** | 54.69 | **Word fMRI** | 56.02 |
| **Diagnostic NLI**| 54.48 | **fMRI** | 10.55 |

##  快速加载

```python
from transformers import AutoTokenizer, AutoModelForMaskedLM

tokenizer = AutoTokenizer.from_pretrained("LuRr/babylm-chinese-deberta-v2-14m-CL")
model = AutoModelForMaskedLM.from_pretrained("LuRr/babylm-chinese-deberta-v2-14m-CL")
```

## 如何复现

克隆本仓库并安装依赖文件：
```bash
pip install -r requirements.txt
```

### 步骤1

执行切分脚本。该过程会自动从 Hugging Face 远程拉取官方 102M 数据集，并按长度物理切分：

```bash
python split_data.py
```
完成后，当前目录下会生成 data_easy、data_medium、data_hard、data_full 四个文件夹。

### 步骤2

执行分词器构建脚本。该过程将训练出大小为 12,800 的词表，并自动完成底层的字符映射替换：

```bash
python train_tokenizer.py
```

### 步骤3

确认 config.json 存在于当前目录后，启动核心训练脚本：

```bash
python cl_train_lr5e-4_epoch30.py
```

### 步骤4

在官方的评测pipeline，将本仓库的配置文件myconfig.yaml应用到测评中，执行：

```bash
python pipeline.py eval --config configs/myconfig.yaml
```
