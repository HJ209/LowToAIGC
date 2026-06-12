"""本地困惑度(perplexity)打分器。

朱雀这类检测器的核心信号之一是"token 可预测性/困惑度"：文本在语言模型下的
平均负对数似然越低（越好预测），越像 AI 生成。我们用一个固定的参考语言模型对
文本打 perplexity，作为"AI 味"的本地代理指标——它不能完全等价于朱雀，但可以在
不消耗朱雀额度的前提下，快速、可复现地横向比较多份文本/多组采样参数。

用法:
    from perplexity import PerplexityScorer
    scorer = PerplexityScorer("Qwen/Qwen2.5-0.5B")
    print(scorer.score("一段中文文本……"))
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class PerplexityResult:
    text_len_tokens: int
    mean_nll: float          # 平均负对数似然（越低越可预测/越像AI）
    perplexity: float        # exp(mean_nll)


class PerplexityScorer:
    """用一个参考因果语言模型计算文本困惑度。

    注意：打分模型固定不变，所有被比较的文本都用同一个模型打分，
    保证横向比较有意义（绝对值不重要，相对高低才重要）。
    """

    def __init__(self, model_name: str = "Qwen/Qwen2.5-0.5B", device: str = "cpu",
                 dtype: torch.dtype = torch.float32):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=dtype
        ).to(device)
        self.model.eval()

    @torch.no_grad()
    def score(self, text: str, max_tokens: int = 2048) -> PerplexityResult:
        ids = self.tokenizer(text, return_tensors="pt").input_ids[:, :max_tokens]
        ids = ids.to(self.device)
        if ids.shape[1] < 2:
            return PerplexityResult(text_len_tokens=int(ids.shape[1]), mean_nll=0.0, perplexity=1.0)
        out = self.model(ids, labels=ids)
        # HuggingFace 默认对 shift 后的 token 取平均交叉熵，即 mean NLL
        mean_nll = float(out.loss.item())
        return PerplexityResult(
            text_len_tokens=int(ids.shape[1]),
            mean_nll=mean_nll,
            perplexity=float(math.exp(mean_nll)),
        )


if __name__ == "__main__":
    import sys

    model = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-0.5B"
    scorer = PerplexityScorer(model)
    for path in sys.argv[2:]:
        with open(path, encoding="utf-8") as f:
            txt = f.read()
        r = scorer.score(txt)
        print(f"{path}\tppl={r.perplexity:8.2f}\tmean_nll={r.mean_nll:.4f}\ttokens={r.text_len_tokens}")
