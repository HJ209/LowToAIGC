"""第2层：用"真人网文"语料 LoRA 微调小模型，从源头降低 AI 指纹。

原理：朱雀主要针对 GPT/Claude/DeepSeek/混元等主流大模型的输出分布训练。把一个
冷门小模型用大量"真人写作"语料做 LoRA 微调，会让它的输出分布向人类靠拢，且对朱雀
而言是"分布外"样本，更难被识别。这是比"事后改写/调采样"更治本的方向。

数据格式（experiments/corpus/human_novels.jsonl，每行一个 JSON）：
    {"text": "一整段真人写的网文正文……"}

硬件：
- 正式训练需要 GPU（>=12GB 显存跑 0.5B~1.5B 较轻松）。
- 本机无 GPU：用 `--allow_cpu --max_steps 20 --model Qwen/Qwen2.5-0.5B` 可在 CPU 上
  做"烟雾测试"——跑通"加载→挂LoRA→训练几步→保存adapter→base vs LoRA 生成+PPL对比"，
  证明整条链路可用、可迁移到 GPU。CPU 上不要指望真实降AI收益，那需要足量真人语料 + GPU。

运行：
    # GPU 正式训练
    python src/finetune_lora.py --model Qwen/Qwen2.5-1.5B --data experiments/corpus/human_novels.jsonl
    # CPU 烟雾测试
    python src/finetune_lora.py --allow_cpu --max_steps 20 --model Qwen/Qwen2.5-0.5B
"""

from __future__ import annotations

import argparse
import os
import sys


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    p.add_argument("--data", default="experiments/corpus/human_novels.jsonl")
    p.add_argument("--out", default="experiments/results/lora_human")
    p.add_argument("--epochs", type=float, default=2.0)
    p.add_argument("--max_steps", type=int, default=-1, help=">0 时覆盖 epochs，用于烟雾测试")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--max_len", type=int, default=1024)
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--allow_cpu", action="store_true", help="允许在无GPU时跑（仅烟雾测试用）")
    p.add_argument("--eval_prompt", default="林远盯着花盆里那颗刚种下的种子，")
    return p.parse_args()


def main():
    args = parse_args()

    import torch

    on_gpu = torch.cuda.is_available()
    if not on_gpu and not args.allow_cpu:
        print(
            "[!] 未检测到 GPU。LoRA 正式训练在 CPU 上不现实（会非常慢）。\n"
            "    请在带 GPU 的机器/Colab/云上运行；或加 --allow_cpu 做小规模烟雾测试。",
            file=sys.stderr,
        )
        sys.exit(2)

    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
    )

    if not os.path.exists(args.data):
        print(f"[!] 找不到语料 {args.data}。请准备真人网文语料（jsonl，每行 {{'text': ...}}）。",
              file=sys.stderr)
        sys.exit(1)

    dtype = torch.bfloat16 if on_gpu else torch.float32
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print(f"[+] 加载 {args.model} (dtype={dtype}, gpu={on_gpu}) ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=dtype, device_map="auto" if on_gpu else None
    )
    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    ds = load_dataset("json", data_files=args.data, split="train")

    def tok_fn(ex):
        return tok(ex["text"], truncation=True, max_length=args.max_len)

    ds = ds.map(tok_fn, remove_columns=ds.column_names)
    collator = DataCollatorForLanguageModeling(tok, mlm=False)

    targs = TrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        bf16=on_gpu,
        logging_steps=5,
        save_strategy="no",
        report_to=[],
    )
    trainer = Trainer(model=model, args=targs, train_dataset=ds, data_collator=collator)
    trainer.train()
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    print(f"[done] LoRA adapter 已保存到 {args.out}", flush=True)

    # ---- 评估：同一 prompt 下 base vs LoRA 生成 + 困惑度对比 ----
    _evaluate(model, tok, args.eval_prompt)


def _evaluate(peft_model, tok, prompt: str):
    """用同一 prompt，对比'关掉adapter(=base)'与'开着adapter(=LoRA)'的生成与困惑度。"""
    import torch

    from perplexity import PerplexityScorer

    peft_model.eval()
    inputs = tok(prompt, return_tensors="pt").to(peft_model.device)
    gen_kwargs = dict(do_sample=True, temperature=1.0, top_p=0.95,
                      repetition_penalty=1.2, max_new_tokens=160,
                      pad_token_id=tok.eos_token_id)

    with torch.no_grad():
        with peft_model.disable_adapter():
            base_ids = peft_model.generate(**inputs, **gen_kwargs)
        lora_ids = peft_model.generate(**inputs, **gen_kwargs)

    base_txt = tok.decode(base_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    lora_txt = tok.decode(lora_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

    scorer = PerplexityScorer("Qwen/Qwen2.5-0.5B")
    print("\n==== base（未微调）生成 ====\n" + base_txt)
    print(f"  PPL={scorer.score(base_txt).perplexity:.2f}")
    print("\n==== LoRA（微调后）生成 ====\n" + lora_txt)
    print(f"  PPL={scorer.score(lora_txt).perplexity:.2f}")


if __name__ == "__main__":
    main()
