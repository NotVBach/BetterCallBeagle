import json, os, sys

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.join(ROOT, "train/base")
OUT = os.path.join(ROOT, "train/merged")
MAX_LEN = 1024
END = "<|im_end|>\n"


def load(path):
    if not os.path.exists(path):
        sys.exit(f"missing {path}")
    return [json.loads(line) for line in open(path, encoding="utf-8")]


def encode(tok, msgs):
    prompt = tok.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True)
    p = tok(prompt, add_special_tokens=False)["input_ids"]
    a = tok(msgs[-1]["content"] + END, add_special_tokens=False)["input_ids"]
    return {"input_ids": (p + a)[:MAX_LEN], "labels": ([-100] * len(p) + a)[:MAX_LEN]}


def collate(pad_id):
    def fn(batch):
        n = max(len(b["input_ids"]) for b in batch)
        pad = lambda xs, v: [x + [v] * (n - len(x)) for x in xs]  # noqa: E731
        ids = [b["input_ids"] for b in batch]
        return {"input_ids": torch.tensor(pad(ids, pad_id)),
                "labels": torch.tensor(pad([b["labels"] for b in batch], -100)),
                "attention_mask": torch.tensor(pad([[1] * len(x) for x in ids], 0))}
    return fn


def main():
    if not os.path.isdir(BASE):
        sys.exit(f"missing {BASE}")
    tok = AutoTokenizer.from_pretrained(BASE, local_files_only=True)
    train = [encode(tok, r["messages"]) for r in load(os.path.join(ROOT, "data/train.jsonl"))]
    evals = [encode(tok, r["messages"]) for r in load(os.path.join(ROOT, "data/eval.jsonl"))]

    # self-check: labels cover exactly the answer + end token
    first = load(os.path.join(ROOT, "data/train.jsonl"))[0]["messages"][-1]["content"] + END
    got = tok.decode([t for t in train[0]["labels"] if t != -100])
    assert got == first or len(train[0]["input_ids"]) == MAX_LEN, f"label mask wrong: {got[:80]!r}"
    print(f"train {len(train)}, eval {len(evals)}")

    model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16, local_files_only=True)
    model.enable_input_require_grads()  # needed for gradient checkpointing with frozen base weights
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                                             target_modules="all-linear", task_type="CAUSAL_LM"))
    model.print_trainable_parameters()

    targs = TrainingArguments(
        output_dir=os.path.join(ROOT, "train/ckpt"), num_train_epochs=2, learning_rate=2e-4,
        per_device_train_batch_size=4, per_device_eval_batch_size=4, gradient_accumulation_steps=4,
        lr_scheduler_type="cosine", warmup_ratio=0.03, bf16=True, logging_steps=20,
        eval_strategy="epoch", save_strategy="no", report_to="none", remove_unused_columns=False,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False})
    trainer = Trainer(model=model, args=targs, train_dataset=train, eval_dataset=evals,
                      data_collator=collate(tok.pad_token_id))
    before = trainer.evaluate()["eval_loss"]
    print(f"eval loss before: {before:.4f}")
    trainer.train()
    after = trainer.evaluate()["eval_loss"]
    print(f"eval loss after:  {after:.4f}  ({'better' if after < before else 'NOT better'})")

    merged = model.merge_and_unload()
    merged.save_pretrained(OUT, safe_serialization=True)
    tok.save_pretrained(OUT)
    print(f"saved {OUT}; copy it back to the host's train/merged/")


if __name__ == "__main__":
    main()
