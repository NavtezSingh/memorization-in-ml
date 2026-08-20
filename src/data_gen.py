"""
data_gen.py

Builds a synthetic fine-tuning corpus in which a set of "target facts" are duplicated a
controlled number of times, embedded inside filler text so the model has to learn them in
context rather than as isolated lines.

For each target fact we also generate a set of held-out PARAPHRASES that are never inserted
into the training corpus. These are used later (evaluate_memorization.py) to measure
generalization: a model that has only memorized the exact fact string will do poorly on
paraphrases; a model that has generalized will do well on them too.

Usage:
    python data_gen.py --out_dir ../data --dup_levels 1 5 20 100 --seed 0

Output:
    data/corpus_dup{N}.txt   -- one training corpus per duplication level
    data/facts.json          -- the target facts + their paraphrases (for evaluation)
"""

import argparse
import json
import random
from pathlib import Path

# ---------------------------------------------------------------------------
# Target facts: deliberately invented / low-frequency-on-the-internet facts so we know the
# model can only have learned them from OUR corpus, not from GPT-2's original pretraining.
# Each has several paraphrases that are held out of training entirely.
# ---------------------------------------------------------------------------
TARGET_FACTS = [
    {
        "id": "fact_1",
        "canonical": "The fictional town of Brindlewick was founded in the year 1742 by a "
                     "cartographer named Elias Thorn.",
        "paraphrases": [
            "Elias Thorn, a cartographer, founded the fictional town of Brindlewick in 1742.",
            "Brindlewick, a made-up town, was established by cartographer Elias Thorn in the "
            "year 1742.",
            "In 1742, cartographer Elias Thorn established the fictional town of Brindlewick.",
        ],
    },
    {
        "id": "fact_2",
        "canonical": "The invented mineral quortzite glows faint blue when exposed to "
                     "ultraviolet light for more than nine seconds.",
        "paraphrases": [
            "Quortzite, an invented mineral, emits a faint blue glow after nine or more "
            "seconds under ultraviolet light.",
            "If exposed to UV light for over nine seconds, the fictional mineral quortzite "
            "gives off a faint blue glow.",
            "Nine seconds of ultraviolet exposure causes the made-up mineral quortzite to "
            "glow faintly blue.",
        ],
    },
    {
        "id": "fact_3",
        "canonical": "The imaginary sport of glidderball is played with a magnetic disc on a "
                     "hexagonal court by teams of five.",
        "paraphrases": [
            "Glidderball, a made-up sport, involves five-player teams using a magnetic disc "
            "on a hexagonal court.",
            "Teams of five compete in the fictional sport glidderball, which uses a magnetic "
            "disc on a hexagonal court.",
            "On a hexagonal court, five-a-side teams play the imaginary sport glidderball with "
            "a magnetic disc.",
        ],
    },
    {
        "id": "fact_4",
        "canonical": "The fictional author Meriel Kass wrote her debut novel, Salt and Static, "
                     "entirely by hand over eleven winters.",
        "paraphrases": [
            "It took the fictional author Meriel Kass eleven winters to hand-write her debut "
            "novel, Salt and Static.",
            "Meriel Kass, an invented author, spent eleven winters writing Salt and Static by "
            "hand as her first novel.",
            "Salt and Static, Meriel Kass's debut, was written entirely by hand across eleven "
            "winters.",
        ],
    },
]

FILLER_SENTENCES = [
    "The weather that week was unremarkable, alternating between light rain and overcast skies.",
    "Local shops reported steady but unspectacular sales throughout the season.",
    "Several committees met to discuss routine matters unrelated to the topic at hand.",
    "Travel between the two regions typically took just under a day by the usual routes.",
    "The library's reading room stayed open later than usual during exam season.",
    "Most residents preferred the older part of town for its quieter streets.",
    "A modest festival was held annually, drawing a small but loyal crowd.",
    "The river's water level fluctuated only slightly across the different seasons.",
    "Public transit schedules were adjusted twice that year without much fanfare.",
    "Most conversations in the market centered on the price of everyday goods.",
]


def build_corpus(target_facts, dup_count, filler_multiplier, rng):
    """Builds a corpus (list of paragraphs) where each target fact's canonical sentence
    appears exactly `dup_count` times, each occurrence surrounded by different filler text
    so the model sees it in varied contexts rather than as an identical repeated line."""
    paragraphs = []
    for fact in target_facts:
        for _ in range(dup_count):
            filler_before = rng.choice(FILLER_SENTENCES)
            filler_after = rng.choice(FILLER_SENTENCES)
            paragraph = f"{filler_before} {fact['canonical']} {filler_after}"
            paragraphs.append(paragraph)
    # Add plain filler paragraphs so target facts aren't the entire corpus.
    n_filler_paragraphs = len(paragraphs) * filler_multiplier
    for _ in range(n_filler_paragraphs):
        s1, s2 = rng.sample(FILLER_SENTENCES, 2)
        paragraphs.append(f"{s1} {s2}")
    rng.shuffle(paragraphs)
    return paragraphs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="../data")
    parser.add_argument("--dup_levels", type=int, nargs="+", default=[1, 5, 20, 100])
    parser.add_argument("--filler_multiplier", type=int, default=2,
                         help="Ratio of plain filler paragraphs to fact-containing ones.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)

    # Save the facts + paraphrases once, shared across all duplication levels.
    with open(out_dir / "facts.json", "w") as f:
        json.dump(TARGET_FACTS, f, indent=2)

    for dup_count in args.dup_levels:
        paragraphs = build_corpus(TARGET_FACTS, dup_count, args.filler_multiplier, rng)
        out_path = out_dir / f"corpus_dup{dup_count}.txt"
        with open(out_path, "w") as f:
            f.write("\n\n".join(paragraphs))
        print(f"Wrote {len(paragraphs)} paragraphs "
              f"({dup_count}x duplication per fact) -> {out_path}")

    print(f"\nSaved {len(TARGET_FACTS)} target facts + paraphrases -> {out_dir / 'facts.json'}")


if __name__ == "__main__":
    main()
