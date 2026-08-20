"""
evaluate_memorization.py

For each fine-tuned checkpoint (one per duplication level), measures:

1. MEMORIZATION — prompt the model with the first ~half of each canonical fact sentence and
   check whether it completes the rest verbatim (or near-verbatim, via token overlap).
2. GENERALIZATION — compute the model's perplexity on the held-out PARAPHRASES of each fact
   (sentences it never saw during training). Lower perplexity = better generalization.

Produces a plot: duplication count (x-axis) vs. memorization rate and paraphrase perplexity
(two lines, y-axis), saved to outputs/memorization_vs_generalization.png

Usage:
    python evaluate_memorization.py \
        --ckpt_dir_pattern "../outputs/ckpt_dup{n}" \
        --dup_levels 1 5 20 100 \
        --facts_json ../data/facts.json \
        --out_dir ../outputs
"""

import argparse
import json
from pathlib import Path

import torch
import matplotlib.pyplot as plt
from transformers import GPT2LMHeadModel, GPT2TokenizerFast


@torch.no_grad()
def completion_overlap_score(model, tokenizer, canonical_sentence, device, prompt_frac=0.3):
    """Feeds the model only the first `prompt_frac` of the sentence (by words -- deliberately
    small, so most of the sentence must be recalled rather than guessed from context) and
    greedily generates a continuation. Returns a CONTINUOUS overlap score in [0, 1] (fraction
    of true-continuation words reproduced in the right position) rather than a binary
    threshold, so results have resolution finer than 1/num_facts. Average this across facts
    to get a memorization score for a checkpoint; near 1.0 means near-verbatim recall, near 0
    means no recall."""
    words = canonical_sentence.split()
    split_idx = max(1, int(len(words) * prompt_frac))
    prompt = " ".join(words[:split_idx])
    true_continuation = " ".join(words[split_idx:])

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    gen_len = len(tokenizer(true_continuation).input_ids) + 5

    output = model.generate(
        input_ids,
        max_new_tokens=gen_len,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    generated = tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True)

    true_words = true_continuation.lower().split()
    gen_words = generated.lower().split()
    if not true_words:
        return 0.0
    # Position-sensitive overlap: reward getting words right in roughly the right place,
    # not just anywhere in the continuation (a pure set-overlap can be inflated by common
    # words like "the"/"a" appearing anywhere).
    matches = sum(1 for i, w in enumerate(true_words) if i < len(gen_words) and gen_words[i] == w)
    return matches / len(true_words)


@torch.no_grad()
def sentence_perplexity(model, tokenizer, sentence, device):
    input_ids = tokenizer(sentence, return_tensors="pt").input_ids.to(device)
    outputs = model(input_ids, labels=input_ids)
    return torch.exp(outputs.loss).item()


def evaluate_checkpoint(ckpt_dir, facts, device):
    tokenizer = GPT2TokenizerFast.from_pretrained(ckpt_dir)
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained(ckpt_dir).to(device).eval()

    overlap_scores = []
    paraphrase_ppls = []

    for fact in facts:
        overlap_scores.append(
            completion_overlap_score(model, tokenizer, fact["canonical"], device)
        )
        for paraphrase in fact["paraphrases"]:
            paraphrase_ppls.append(sentence_perplexity(model, tokenizer, paraphrase, device))

    memorization_score = sum(overlap_scores) / len(overlap_scores)
    avg_paraphrase_ppl = sum(paraphrase_ppls) / len(paraphrase_ppls)
    return memorization_score, avg_paraphrase_ppl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir_pattern", type=str, default="../outputs/ckpt_dup{n}",
                         help="Use {n} as a placeholder for the duplication level.")
    parser.add_argument("--dup_levels", type=int, nargs="+", default=[1, 5, 20, 100])
    parser.add_argument("--facts_json", type=str, default="../data/facts.json")
    parser.add_argument("--out_dir", type=str, default="../outputs")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    facts = json.loads(Path(args.facts_json).read_text())

    results = []
    for n in args.dup_levels:
        ckpt_dir = args.ckpt_dir_pattern.format(n=n)
        print(f"Evaluating checkpoint for duplication level {n} ({ckpt_dir}) ...")
        mem_rate, ppl = evaluate_checkpoint(ckpt_dir, facts, device)
        print(f"  memorization_score={mem_rate:.2f}  paraphrase_perplexity={ppl:.2f}")
        results.append({"dup_level": n, "memorization_score": mem_rate,
                         "paraphrase_perplexity": ppl})

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "memorization_results.json", "w") as f:
        json.dump(results, f, indent=2)

    dup_levels = [r["dup_level"] for r in results]
    mem_rates = [r["memorization_score"] for r in results]
    ppls = [r["paraphrase_perplexity"] for r in results]

    fig, ax1 = plt.subplots(figsize=(7, 5))
    color1 = "tab:blue"
    ax1.set_xlabel("Duplication count (per fact)")
    ax1.set_ylabel("Memorization score (positional word overlap)", color=color1)
    ax1.plot(dup_levels, mem_rates, marker="o", color=color1, label="Memorization score")
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.set_xscale("log")

    ax2 = ax1.twinx()
    color2 = "tab:red"
    ax2.set_ylabel("Avg. perplexity on held-out paraphrases", color=color2)
    ax2.plot(dup_levels, ppls, marker="s", color=color2, label="Paraphrase perplexity")
    ax2.tick_params(axis="y", labelcolor=color2)

    plt.title("Memorization vs. Generalization as Duplication Increases")
    fig.tight_layout()
    plot_path = out_dir / "memorization_vs_generalization.png"
    plt.savefig(plot_path, dpi=150)
    print(f"\nSaved plot -> {plot_path}")
    print(f"Saved raw results -> {out_dir / 'memorization_results.json'}")


if __name__ == "__main__":
    main()
