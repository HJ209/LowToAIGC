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

## 目录结构

```
src/
  perplexity.py           # 用参考LM给文本打困惑度（AI味代理指标）
  sampling_experiment.py  # 第1层：采样参数 -> 困惑度对照实验
  finetune_lora.py        # 第2层：真人语料 LoRA 微调脚手架（需GPU）
experiments/
  samples/                # 原版与各改写版样本（已附朱雀实测结果）
  results/                # 实验输出（困惑度对照表、生成样本）
reports/
  zhuque_lower_ai_rate_report.md  # 朱雀检测原理与降AI方法完整报告
```

## 快速开始

```bash
pip install -r requirements.txt

# 给已有文本打困惑度（越高越不像AI）
python src/perplexity.py Qwen/Qwen2.5-0.5B experiments/samples/ch3_original.txt

# 第1层：扫采样参数，产出 参数->困惑度 对照表
python src/sampling_experiment.py

# 第2层：真人语料 LoRA 微调（需 GPU + 准备 jsonl 语料）
python src/finetune_lora.py --model Qwen/Qwen2.5-1.5B --data experiments/corpus/human_novels.jsonl
```

## 工作流建议（务实预期）

1. 生成阶段：用微调后的小模型 + 调好的采样参数出初稿（压低困惑度指纹）。
2. 用 `perplexity.py` 本地筛查，挑困惑度最高的几版。
3. 关键段落（对话、心理活动）**真人补写**——这是唯一能把"人类创作"占比做上去的方法。
4. 朱雀仅做最终抽检，别依赖它逐版刷。
