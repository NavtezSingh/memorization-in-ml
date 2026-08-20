"""
evaluate_diversity.py

For each fine-tuned checkpoint, generates several continuations for prompts related to each
target fact (using sampling, not greedy decoding) and measures how DIVERSE those generations
are. The hypothesis: heavier duplication -> the model collapses toward repeating the exact
memorized sentence -> lower diversity. This is the "creativity" angle of the project.

Two diversity metrics:
1. distinct-2 — fraction of unique bigrams across all generations for a fact (classic NLG
   diversity metric, no extra dependencies beyond what we already use).
2. Embedding diversity — average pairwise cosine distance between generation embeddings
   (via sentence-transformers), a more semantic notion of diversity.

Produces: outputs/diversity_vs_duplication.png

Usage:
    python evaluate_diversity.py \
        --ckpt_dir_pattern "../outputs/ckpt_dup{n}" \
        --dup_levels 1 5 20 100 \
        --facts_json ../data/facts.json \
        --out_dir ../outputs \
        --n_samples 8
"""

import argparse
import json
import itertools
from pathlib import Path

import torch
import matplotlib.pyplot as plt
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from sentence_transformers import SentenceTransformer, util


def distinct_n(texts, n=2):
    all_ngrams = []
    for t in texts:
        tokens = t.lower().split()
        all_ngrams.extend(zip(*[tokens[i:] for i in range(n)]))
    if not all_ngrams:
        return 0.0
    return len(set(all_ngrams)) / len(all_ngrams)


@torch.no_grad()
def generate_samples(model, tokenizer, prompt, device, n_samples=8, max_new_tokens=30):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    outputs = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        top_p=0.95,
        temperature=1.0,
        num_return_sequences=n_samples,
        pad_token_id=tokenizer.eos_token_id,
    )
    return [
        tokenizer.decode(o[input_ids.shape[1]:], skip_special_tokens=True) for o in outputs
    ]


def embedding_diversity(embedder, texts):
    if len(texts) < 2:
        return 0.0
    embeddings = embedder.encode(texts, convert_to_tensor=True)
    pairs = list(itertools.combinations(range(len(texts)), 2))
    dists = [1 - util.cos_sim(embeddings[i], embeddings[j]).item() for i, j in pairs]
    return sum(dists) / len(dists)


def evaluate_checkpoint(ckpt_dir, facts, device, embedder, n_samples):
    tokenizer = GPT2TokenizerFast.from_pretrained(ckpt_dir)
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained(ckpt_dir).to(device).eval()

    distinct2_scores = []
    embed_div_scores = []

    for fact in facts:
        # Prompt with just the first few words to see what the model does when it has to
        # "choose" how to continue -- does it collapse to the memorized sentence every time?
        prompt = " ".join(fact["canonical"].split()[:4])
        samples = generate_samples(model, tokenizer, prompt, device, n_samples=n_samples)
        distinct2_scores.append(distinct_n(samples, n=2))
        embed_div_scores.append(embedding_diversity(embedder, samples))

    return (sum(distinct2_scores) / len(distinct2_scores),
            sum(embed_div_scores) / len(embed_div_scores))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir_pattern", type=str, default="../outputs/ckpt_dup{n}")
    parser.add_argument("--dup_levels", type=int, nargs="+", default=[1, 5, 20, 100])
    parser.add_argument("--facts_json", type=str, default="../data/facts.json")
    parser.add_argument("--out_dir", type=str, default="../outputs")
    parser.add_argument("--n_samples", type=int, default=8)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    facts = json.loads(Path(args.facts_json).read_text())

    print("Loading sentence embedder for semantic diversity metric...")
    embedder = SentenceTransformer("all-MiniLM-L6-v2", device=device)

    results = []
    for n in args.dup_levels:
        ckpt_dir = args.ckpt_dir_pattern.format(n=n)
        print(f"Evaluating diversity for duplication level {n} ({ckpt_dir}) ...")
        distinct2, embed_div = evaluate_checkpoint(
            ckpt_dir, facts, device, embedder, args.n_samples
        )
        print(f"  distinct-2={distinct2:.3f}  embedding_diversity={embed_div:.3f}")
        results.append({"dup_level": n, "distinct_2": distinct2,
                         "embedding_diversity": embed_div})

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "diversity_results.json", "w") as f:
        json.dump(results, f, indent=2)

    dup_levels = [r["dup_level"] for r in results]
    distinct2s = [r["distinct_2"] for r in results]
    embed_divs = [r["embedding_diversity"] for r in results]

    fig, ax1 = plt.subplots(figsize=(7, 5))
    color1 = "tab:green"
    ax1.set_xlabel("Duplication count (per fact)")
    ax1.set_ylabel("distinct-2", color=color1)
    ax1.plot(dup_levels, distinct2s, marker="o", color=color1, label="distinct-2")
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.set_xscale("log")

    ax2 = ax1.twinx()
    color2 = "tab:purple"
    ax2.set_ylabel("Embedding diversity (avg. pairwise cosine distance)", color=color2)
    ax2.plot(dup_levels, embed_divs, marker="s", color=color2, label="Embedding diversity")
    ax2.tick_params(axis="y", labelcolor=color2)

    plt.title("Generation Diversity as Duplication Increases")
    fig.tight_layout()
    plot_path = out_dir / "diversity_vs_duplication.png"
    plt.savefig(plot_path, dpi=150)
    print(f"\nSaved plot -> {plot_path}")
    print(f"Saved raw results -> {out_dir / 'diversity_results.json'}")


if __name__ == "__main__":
    main()
