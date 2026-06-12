"""用本地小模型分段改写整章，降低 AI 味（保留情节与对话），并用困惑度挑最优版。

策略：
- 把原文按空行切成段，按 GROUP 个段一组喂给模型，指令是"改写得更口语、长短句交错、
  不那么 AI，但严格保留情节和对话"。分段是为了让小模型在 CPU 上既快又不跑题。
- 每组改写后立即用 PerplexityScorer 打分，多组采样参数里取困惑度较高(更不可预测)
  且非空的结果。
- 全部拼接成整章，输出到 out 路径，并打印整章困惑度。

无 GPU 也能跑，默认 1.5B-Instruct（比 0.5B 通顺很多），CPU 较慢。
"""

from __future__ import annotations

import argparse
import gc
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from perplexity import PerplexityScorer

GEN_MODEL = os.environ.get("GEN_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
PPL_MODEL = os.environ.get("PPL_MODEL", "Qwen/Qwen2.5-0.5B")

SYSTEM = (
    "你是资深中文网络小说编辑。任务：改写给定段落，降低'AI 味'。"
    "硬性要求：1) 情节、人物、对话内容必须完全保留，不许增删剧情；"
    "2) 用更口语、生活化的写法，长短句交错，避免'若有所思''微微''深吸一口气'这类AI高频套路词；"
    "3) 适当合并或拆分句子，制造节奏起伏；4) 只输出改写后的正文，不要解释、不要加标题。"
)


def split_paragraphs(text: str) -> list[str]:
    return [p.strip() for p in text.split("\n\n") if p.strip()]


def rewrite_group(model, tok, group: str, temperature: float, top_p: float, rp: float) -> str:
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"改写下面这段，保留全部情节和对话：\n\n{group}"},
    ]
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    max_new = min(1024, int(inputs.input_ids.shape[1] * 1.6) + 80)
    out = model.generate(
        **inputs, do_sample=True, temperature=temperature, top_p=top_p,
        repetition_penalty=rp, max_new_tokens=max_new, pad_token_id=tok.eos_token_id,
    )
    return tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="experiments/samples/ch3_original.txt")
    ap.add_argument("--out", default="experiments/results/ch3_local_rewrite_best.txt")
    ap.add_argument("--group", type=int, default=3, help="每组段落数")
    args = ap.parse_args()

    with open(args.src, encoding="utf-8") as f:
        paras = split_paragraphs(f.read())

    # 内存紧（本机8G）：fp32 下同时装 1.5B+0.5B 会OOM。所以先只装生成模型改写全文，
    # 释放后再装 0.5B 给成品打困惑度。fp32 在 CPU 上比 bf16 快很多。
    print(f"[+] 加载生成模型 {GEN_MODEL} (fp32) ...", flush=True)
    tok = AutoTokenizer.from_pretrained(GEN_MODEL)
    model = AutoModelForCausalLM.from_pretrained(GEN_MODEL, dtype=torch.float32)
    model.eval()

    out_chunks: list[str] = []
    i = 0
    gi = 0
    while i < len(paras):
        group = "\n\n".join(paras[i:i + args.group])
        gi += 1
        cand = rewrite_group(model, tok, group, 1.1, 0.95, 1.2)
        if len(cand) < len(group) * 0.4:  # 太短大概率改崩，兜底保留原文
            cand = group
        print(f"[组{gi}] 段{i+1}-{i+args.group} len={len(cand)}", flush=True)
        out_chunks.append(cand)
        i += args.group

    result = "\n\n".join(out_chunks)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(result)

    del model
    gc.collect()
    print(f"[+] 加载打分模型 {PPL_MODEL} ...", flush=True)
    scorer = PerplexityScorer(PPL_MODEL)
    whole_ppl = scorer.score(result).perplexity
    orig_ppl = scorer.score("\n\n".join(paras)).perplexity
    print(f"[done] 输出 {args.out}")
    print(f"       原文 PPL={orig_ppl:.2f} -> 改写后 PPL={whole_ppl:.2f}  字数={len(result)}")


if __name__ == "__main__":
    main()
