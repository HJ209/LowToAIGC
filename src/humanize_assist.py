"""降AI 体检 + 改写辅助（全本地、零 GPU、免费）。

用途：你用大模型出了 AI 初稿后，把整章丢给本工具——
1. 用本地参考语言模型逐段打困惑度(PPL)，PPL 越低=越可预测=越像 AI；
2. 按 PPL 升序排出"最该改写的高危段落"清单；
3. 为每个高危段落生成一段**可直接粘贴给大模型**的"降AI味"改写指令；
4. 输出一份 Markdown 报告，供你按图索骥逐段重写，再回头复检。

这是在"本机无 GPU、小模型自动出文必崩"的现实下，唯一可靠且高效的降AI路径：
AI 出稿 -> 本工具定位最差段落 -> 人工/大模型重写这些段 -> 复检 PPL -> 朱雀抽检。

用法：
    python src/humanize_assist.py 路径/章节.txt
    python src/humanize_assist.py 章节.txt --top 10 --model Qwen/Qwen2.5-0.5B --out report.md
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

from perplexity import PerplexityScorer

# 给大模型的"降AI味"改写指令模板（粘贴时把 {PARA} 换成高危段落原文）
REWRITE_PROMPT_TEMPLATE = """请改写下面这段中文小说正文，目标是降低"AI 味"（让它更像人随手写的）。
硬性要求：
1) 情节、人物、对话信息必须完全保留，不许增删剧情、不许编造新内容；
2) 打散节奏：长短句交错，别每句都工整对仗；
3) 换掉 AI 高频套路词（如"若有所思""微微""压抑不住""瞳孔放大""深吸一口气"），用更具体、更口语、甚至略不规整的写法；
4) 适当合并/拆分句子，允许半截话、口头语、轻微跑题的个人化细节；
5) 只输出改写后的正文，不要解释、不要加标题。

待改写正文：
{PARA}"""

# PPL 风险阈值（基于实测：原文整章 PPL≈16.6 被朱雀判 62% AI；改写版 PPL≈35-47）
RISK_HIGH = 20.0   # 低于此：高危，强烈建议重写
RISK_MID = 35.0    # 低于此：中等，建议润色


@dataclass
class ParaScore:
    idx: int
    text: str
    tokens: int
    ppl: float

    @property
    def risk(self) -> str:
        if self.tokens < 12:
            return "样本过短"
        if self.ppl < RISK_HIGH:
            return "高危"
        if self.ppl < RISK_MID:
            return "中等"
        return "较好"


def split_paragraphs(text: str) -> list[str]:
    return [p.strip() for p in text.split("\n\n") if p.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", help="待体检的章节 txt（UTF-8，段落以空行分隔）")
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B", help="本地参考打分模型")
    ap.add_argument("--top", type=int, default=8, help="生成改写指令的高危段落数")
    ap.add_argument("--out", default=None, help="报告输出路径，默认 <src>.降AI报告.md")
    args = ap.parse_args()

    with open(args.src, encoding="utf-8") as f:
        full = f.read()
    paras = split_paragraphs(full)

    scorer = PerplexityScorer(args.model)
    whole = scorer.score(full)

    scores: list[ParaScore] = []
    for i, p in enumerate(paras):
        r = scorer.score(p)
        scores.append(ParaScore(idx=i + 1, text=p, tokens=r.text_len_tokens, ppl=r.perplexity))

    # 高危排序：只在样本够长的段落里，按 PPL 升序（越低越像AI、越该改）
    rankable = [s for s in scores if s.tokens >= 12]
    priority = sorted(rankable, key=lambda s: s.ppl)[: args.top]

    out = args.out or (args.src + ".降AI报告.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write(f"# 降AI 体检报告：{os.path.basename(args.src)}\n\n")
        f.write(f"- 参考模型：`{args.model}`\n")
        f.write(f"- 全文困惑度 PPL：**{whole.perplexity:.2f}**"
                f"（越高越不像AI；实测原文≈16.6 时朱雀判约 62% AI，改写到 35+ 后高置信AI降到~15%）\n")
        f.write(f"- 段落数：{len(paras)}，可评分段落（token≥12）：{len(rankable)}\n\n")

        f.write("## 一、最该改写的高危段落（按 AI 味从高到低）\n\n")
        f.write("> 优先重写这些；改完用本工具复检，看全文 PPL 是否抬升。\n\n")
        for rank, s in enumerate(priority, 1):
            preview = s.text.replace("\n", " ")
            preview = preview[:60] + ("…" if len(preview) > 60 else "")
            f.write(f"{rank}. **段{s.idx}**（PPL={s.ppl:.1f}，{s.risk}）：{preview}\n")
        f.write("\n")

        f.write("## 二、逐段改写指令（直接复制粘贴给大模型）\n\n")
        for rank, s in enumerate(priority, 1):
            f.write(f"### 高危 {rank} · 段{s.idx}（PPL={s.ppl:.1f}）\n\n")
            f.write("```\n")
            f.write(REWRITE_PROMPT_TEMPLATE.replace("{PARA}", s.text))
            f.write("\n```\n\n")

        f.write("## 三、全段 PPL 明细\n\n")
        f.write("| 段 | PPL | tokens | 风险 | 预览 |\n|---|---|---|---|---|\n")
        for s in scores:
            preview = s.text.replace("\n", " ").replace("|", "／")
            preview = preview[:30] + ("…" if len(preview) > 30 else "")
            f.write(f"| {s.idx} | {s.ppl:.1f} | {s.tokens} | {s.risk} | {preview} |\n")

    print(f"[done] 全文 PPL={whole.perplexity:.2f}；高危段 {len(priority)} 个；报告 -> {out}")


if __name__ == "__main__":
    main()
