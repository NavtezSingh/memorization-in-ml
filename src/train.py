"""
train.py

Fine-tunes GPT-2 small (or any small causal LM) on one corpus at a time, saving a checkpoint
per duplication level so evaluate_memorization.py / evaluate_diversity.py can compare across
levels.

Usage (run once per duplication level, or loop over all of them):
    python train.py --corpus ../data/corpus_dup20.txt --out_dir ../outputs/ckpt_dup20 \
        --epochs 3 --model gpt2

For a quick loop over every generated corpus:
    for f in ../data/corpus_dup*.txt; do
        n=$(basename "$f" .txt | sed 's/corpus_dup//')
        python train.py --corpus "$f" --out_dir "../outputs/ckpt_dup${n}"
    done
"""

import argparse
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import (
    GPT2LMHeadModel,
    GPT2TokenizerFast,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)


class LineByLineTextDataset(Dataset):
    """Simple dataset: tokenizes each paragraph (separated by blank lines in the corpus)
    into a fixed-length block. Good enough for a small synthetic corpus like ours."""

    def __init__(self, tokenizer, file_path, block_size=128):
        text = Path(file_path).read_text()
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        self.examples = []
        for p in paragraphs:
            tokenized = tokenizer(
                p, truncation=True, max_length=block_size, padding="max_length"
            )
            self.examples.append(tokenized["input_ids"])

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return torch.tensor(self.examples[idx], dtype=torch.long)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--model", type=str, default="gpt2",
                         help="HF model name, e.g. gpt2 (124M) or gpt2-medium if you have "
                              "the compute budget.")
    parser.add_argument("--epochs", type=int, default=1,
                         help="Default is 1 (a single pass over the corpus). This is "
                              "intentional: with epochs=1, each occurrence of a fact is seen "
                              "by the optimizer exactly once, so total training exposure to a "
                              "given fact string equals its duplication count exactly -- no "
                              "epoch confound. Avoid setting this higher for this project; "
                              "multiple epochs let even low-duplication facts get memorized, "
                              "erasing the effect you're trying to measure.")
    parser.add_argument("--max_steps", type=int, default=-1,
                         help="Generally NOT recommended for this project -- fixing total "
                              "steps across differently-sized corpora inverts the intended "
                              "manipulation (smaller corpora end up training far more epochs "
                              "than larger ones). Prefer the epochs=1 default instead.")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5,
                         help="Lowered from a more typical 5e-5 to slow down how fast the "
                              "model overfits the tiny corpus.")
    parser.add_argument("--block_size", type=int, default=128)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    tokenizer = GPT2TokenizerFast.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token  # GPT-2 has no pad token by default

    model = GPT2LMHeadModel.from_pretrained(args.model).to(device)

    dataset = LineByLineTextDataset(tokenizer, args.corpus, block_size=args.block_size)
    print(f"Loaded {len(dataset)} training examples from {args.corpus}")

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    training_args_kwargs = dict(
        output_dir=args.out_dir,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        logging_steps=10,
        save_strategy="no",  # we save the final model manually below
        report_to="none",
        fp16=torch.cuda.is_available(),
    )
    if args.max_steps > 0:
        training_args_kwargs["max_steps"] = args.max_steps
    else:
        training_args_kwargs["num_train_epochs"] = args.epochs

    training_args = TrainingArguments(**training_args_kwargs)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
    )

    trainer.train()

    trainer.save_model(args.out_dir)
    tokenizer.save_pretrained(args.out_dir)
    print(f"Saved fine-tuned model -> {args.out_dir}")


if __name__ == "__main__":
    main()
