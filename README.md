# LowToAIGC — 降低 AI 生成文本的检测率（针对腾讯朱雀）

研究并工程化"如何降低 AI 生成中文文本被检测为 AI 的概率"，检测基准为腾讯朱雀
（https://matrix.tencent.com/ai-detect/）。本仓库同时包含**分析报告**、**可复现实验**
和**生成阶段降 AI 的代码方案**。

> 仅用于个人创作（如网络小说）风格研究，请勿用于学术不端或规避任何合规标识义务。
> 任何方法都不保证 100% 通过检测，朱雀模型持续更新。

## 核心结论（先看这个）

朱雀判定的不是"读起来像不像人"，而是"文本是不是模型生成的统计指纹"，三大维度：
**① token 可预测性/困惑度** ② 句子节奏突发性 ③ 段落/全文风格一致性。

实测一条第3章网文（约5000字）四个版本：

| 文本 | 朱雀·高置信AI | 朱雀·疑似AI | 朱雀·人类 | 本地困惑度PPL |
|---|---|---|---|---|
| 原版（纯AI） | 62.74% | 37.26% | 0% | 16.65 |
| 轻度改写（换词+口语） | 15.74% | 84.26% | 0% | 35.85 |
| 深度改写（打散结构） | 14.95% | 85.05% | 0% | 47.39 |
| 混排（插"真人式"段） | 22.31% | 77.69% | 0% | 51.40 |

要点：
- **事后改写能把"高置信 AI"从 60%+ 压到 ~15%**（提高困惑度有效），但**顶不起"人类创作"占比**——因为只要是 AI 写的（哪怕模仿人腔），仍带指纹。
- **本地困惑度(PPL)是朱雀高置信AI率的有效免费代理**：PPL 越低 → 朱雀判AI越狠，趋势吻合。可用它在不烧朱雀额度的前提下横扫参数。
- **真正治本在生成阶段**：调采样参数（治标）+ 用真人语料 LoRA 微调小模型（治本，且对朱雀是分布外样本）。

详见 [`reports/zhuque_lower_ai_rate_report.md`](reports/zhuque_lower_ai_rate_report.md)。

## 三层降 AI 方案

| 层 | 方法 | 脚本 | 算力 | 效果 |
|---|---|---|---|---|
| 1 | 调采样参数(temperature/top_p/repetition_penalty) | `src/sampling_experiment.py` | CPU 可 | 治标，压高置信AI |
| 2 | 真人语料 LoRA 微调小模型 | `src/finetune_lora.py` | 需 GPU | 治本，分布外 |
| - | 本地困惑度代理打分（贯穿1/2，免烧朱雀额度） | `src/perplexity.py` | CPU 可 | 评测工具 |
| ⭐ | **降AI 体检+改写辅助**（无GPU时的主力工具） | `src/humanize_assist.py` | CPU 可 | 定位高危段+给改写指令 |

> **无 GPU / 无语料的推荐用法**：第2层微调需要 GPU，本地小模型自动出文质量不可用。
> 所以实际主力是 `humanize_assist.py`：你用大模型出 AI 初稿 → 本工具逐段打分、揪出最像AI的段落、
> 给出可粘贴的降AI改写指令 → 你（或大模型）重写这些段 → 复检。这是零 GPU 下唯一可靠高效的路径。

## 目录结构

```
src/
  perplexity.py           # 用参考LM给文本打困惑度（AI味代理指标）
  sampling_experiment.py  # 第1层：采样参数 -> 困惑度对照实验
  finetune_lora.py        # 第2层：真人语料 LoRA 微调（GPU正式训练/CPU烟雾测试）
  humanize_assist.py      # 降AI体检+改写辅助：逐段PPL打分+高危清单+改写指令（零GPU）
experiments/
  samples/                # 原版与各改写版样本（已附朱雀实测结果）
  results/                # 实验输出（困惑度对照表、生成样本）
reports/
  zhuque_lower_ai_rate_report.md  # 朱雀检测原理与降AI方法完整报告
```

## 快速开始

```bash
pip install -r requirements.txt

# 【主力】降AI 体检：逐段打分 + 高危段清单 + 可粘贴改写指令
python src/humanize_assist.py 你的章节.txt --top 10

# 给已有文本打困惑度（越高越不像AI）
python src/perplexity.py Qwen/Qwen2.5-0.5B experiments/samples/ch3_original.txt

# 第1层：扫采样参数，产出 参数->困惑度 对照表
python src/sampling_experiment.py

# 第2层：真人语料 LoRA 微调（GPU 正式训练，需准备 jsonl 语料）
python src/finetune_lora.py --model Qwen/Qwen2.5-1.5B --data experiments/corpus/human_novels.jsonl
# 第2层：CPU 烟雾测试（无 GPU 也能跑通链路，验证用，不产生真实收益）
python src/finetune_lora.py --allow_cpu --max_steps 20 --model Qwen/Qwen2.5-0.5B \
  --data experiments/samples/human_corpus.sample.jsonl --batch_size 1 --grad_accum 4 --max_len 256
```

## 工作流建议（无 GPU 现实版，务实预期）

1. 用任意大模型（网页版 ChatGPT/DeepSeek/豆包等）出章节 AI 初稿。
2. `python src/humanize_assist.py 章节.txt` → 拿到「高危段清单 + 每段可粘贴的降AI改写指令」。
3. 把高危段的改写指令逐个粘给大模型重写（或自己手写，效果最稳）。
4. 改完回头 `humanize_assist.py` / `perplexity.py` 复检，看全文 PPL 是否抬升。
5. 最后用朱雀抽检 1 次定稿。关键段落（对话、心理活动）**真人补写**是唯一能把"人类创作"占比做上去的方法。

> 有 GPU + 真人语料后，再上第2层 LoRA 微调，从源头出更难被识别的稿。
