"""第1层：采样参数 -> AI味(困惑度) 对照实验。

同一段写作指令，用同一个小模型、在不同解码参数(temperature/top_p/repetition_penalty)
下各生成一版续写，然后用固定参考模型给每一版打 perplexity。困惑度越高，token 越
不可预测，理论上越不容易被朱雀判为"高置信 AI"。

本脚本产出一张"参数 -> 困惑度"对照表(CSV + Markdown)，用于在不消耗朱雀额度的前提下
快速定位"哪些采样参数能显著抬高不可预测性"。挑出最优的若干组，再用朱雀人工复测坐实。

无 GPU 也能跑（默认 0.5B 模型，CPU 推理），只是慢。
"""

from __future__ import annotations

import csv
import itertools
import os
from dataclasses import dataclass, asdict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from perplexity import PerplexityScorer

DEFAULT_GEN_MODEL = os.environ.get("GEN_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
DEFAULT_PPL_MODEL = os.environ.get("PPL_MODEL", "Qwen/Qwen2.5-0.5B")

# 采样参数网格：可按需扩展
TEMPERATURES = [0.7, 1.0, 1.3]
TOP_PS = [0.95, 0.85]
REPETITION_PENALTIES = [1.0, 1.2]


@dataclass
class RunConfig:
    temperature: float
    top_p: float
    repetition_penalty: float


def build_prompt(seed: str) -> list[dict]:
    return [
        {
            "role": "system",
            "content": "你是一个中文网络小说写手，文风自然、有生活气息，善用长短句交错。",
        },
        {"role": "user", "content": seed},
    ]


def generate(model, tokenizer, messages, cfg: RunConfig, max_new_tokens: int = 220) -> str:
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    out = model.generate(
        **inputs,
        do_sample=True,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        repetition_penalty=cfg.repetition_penalty,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
    )
    gen = out[0][inputs.input_ids.shape[1]:]
    return tokenizer.decode(gen, skip_special_tokens=True)


def main():
    seed = (
        "请续写一段都市异能小说：主角林远刚觉醒一种'种子'天赋，回到家后独自摸索这种能力，"
        "尝试把第二颗种子种进阳台的绿萝里。请写约300字，第三人称，注重心理活动与细节。"
    )

    print(f"[+] 加载生成模型 {DEFAULT_GEN_MODEL} ...")
    tok = AutoTokenizer.from_pretrained(DEFAULT_GEN_MODEL)
    gen_model = AutoModelForCausalLM.from_pretrained(
        DEFAULT_GEN_MODEL, torch_dtype=torch.float32
    )
    gen_model.eval()

    print(f"[+] 加载困惑度参考模型 {DEFAULT_PPL_MODEL} ...")
    scorer = PerplexityScorer(DEFAULT_PPL_MODEL)

    os.makedirs("experiments/results/generations", exist_ok=True)
    rows = []
    combos = list(itertools.product(TEMPERATURES, TOP_PS, REPETITION_PENALTIES))
    for i, (t, p, rp) in enumerate(combos):
        cfg = RunConfig(temperature=t, top_p=p, repetition_penalty=rp)
        print(f"[{i+1}/{len(combos)}] 生成 {cfg} ...")
        txt = generate(gen_model, tok, build_prompt(seed), cfg)
        fname = f"experiments/results/generations/gen_t{t}_p{p}_rp{rp}.txt"
        with open(fname, "w", encoding="utf-8") as f:
            f.write(txt)
        ppl = scorer.score(txt)
        rows.append({**asdict(cfg), "perplexity": round(ppl.perplexity, 2),
                     "mean_nll": round(ppl.mean_nll, 4), "tokens": ppl.text_len_tokens,
                     "file": fname})

    rows.sort(key=lambda r: r["perplexity"], reverse=True)

    csv_path = "experiments/results/sampling_ppl.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    md_path = "experiments/results/sampling_ppl.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# 采样参数 -> 困惑度对照（困惑度越高越不可预测，理论上越不像AI）\n\n")
        f.write(f"- 生成模型: `{DEFAULT_GEN_MODEL}`\n- 打分模型: `{DEFAULT_PPL_MODEL}`\n\n")
        f.write("| temperature | top_p | repetition_penalty | perplexity | mean_nll | tokens |\n")
        f.write("|---|---|---|---|---|---|\n")
        for r in rows:
            f.write(f"| {r['temperature']} | {r['top_p']} | {r['repetition_penalty']} | "
                    f"{r['perplexity']} | {r['mean_nll']} | {r['tokens']} |\n")
    print(f"[done] 结果: {csv_path} / {md_path}")


if __name__ == "__main__":
    main()
