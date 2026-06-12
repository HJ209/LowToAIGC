"""第2层：用"真人网文"语料 LoRA 微调小模型，从源头降低 AI 指纹。

原理：朱雀主要针对 GPT/Claude/DeepSeek/混元等主流大模型的输出分布训练。把一个
冷门小模型用大量"真人写作"语料做 LoRA 微调，会让它的输出分布向人类靠拢，且对朱雀
而言是"分布外"样本，更难被识别。这是比"事后改写/调采样"更治本的方向。

数据格式（experiments/corpus/human_novels.jsonl，每行一个 JSON）：
    {"text": "一整段真人写的网文正文……"}

硬件：LoRA 微调需要 GPU（>=12GB 显存跑 0.5B~1.5B 较轻松）。本机无 GPU，仅作脚手架；
请在带 GPU 的机器/Colab/云上运行。脚本会自动检测 CUDA，没有则给出明确提示。

运行：
    python src/finetune_lora.py --model Qwen/Qwen2.5-1.5B --data experiments/corpus/human_novels.jsonl
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
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--max_len", type=int, default=1024)
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    return p.parse_args()


def main():
    args = parse_args()

    import torch

    if not torch.cuda.is_available():
        print(
            "[!] 未检测到 GPU。LoRA 微调在 CPU 上不现实（会非常慢）。\n"
            "    请在带 GPU 的机器/Colab/云上运行本脚本。脚本与依赖已就绪，可直接迁移。",
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

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="auto")
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
        learning_rate=args.lr,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        report_to=[],
    )
    trainer = Trainer(model=model, args=targs, train_dataset=ds, data_collator=collator)
    trainer.train()
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    print(f"[done] LoRA adapter 已保存到 {args.out}")


if __name__ == "__main__":
    main()
