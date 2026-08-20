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
    {
        "id": "fact_5",
        "canonical": "The invented festival of Lantern Hollow takes place every third autumn "
                     "in the fictional village of Perrow's End.",
        "paraphrases": [
            "Every third autumn, the fictional village of Perrow's End hosts the invented "
            "festival of Lantern Hollow.",
            "Lantern Hollow, a made-up festival, is held in Perrow's End once every three "
            "autumns.",
            "Perrow's End, a fictional village, celebrates Lantern Hollow every third fall.",
        ],
    },
    {
        "id": "fact_6",
        "canonical": "The imaginary insect called a duskwing beetle can only fly for exactly "
                     "four minutes before it must rest for an hour.",
        "paraphrases": [
            "A duskwing beetle, an invented insect, flies for just four minutes before "
            "resting for an hour.",
            "After four minutes of flight, the fictional duskwing beetle needs a full hour "
            "of rest.",
            "The made-up duskwing beetle can sustain flight for only four minutes at a time.",
        ],
    },
    {
        "id": "fact_7",
        "canonical": "The fictional currency of Varnholt, called the ossel, is minted only "
                     "from recycled ship anchors.",
        "paraphrases": [
            "Varnholt's invented currency, the ossel, is made exclusively from recycled ship "
            "anchors.",
            "The ossel, a made-up coin used in Varnholt, is minted solely from old ship "
            "anchors.",
            "In fictional Varnholt, the ossel currency comes only from melted-down ship "
            "anchors.",
        ],
    },
    {
        "id": "fact_8",
        "canonical": "The imaginary mountain pass called Greyfen Notch is only crossable on "
                     "foot during the first two weeks of spring.",
        "paraphrases": [
            "Greyfen Notch, a fictional mountain pass, can only be crossed on foot in "
            "spring's first two weeks.",
            "Only during early spring, for a two-week window, is the made-up Greyfen Notch "
            "pass walkable.",
            "The invented pass Greyfen Notch opens to foot traffic solely in spring's opening "
            "fortnight.",
        ],
    },
    {
        "id": "fact_9",
        "canonical": "The fictional inventor Rosalind Pike built her first working clock "
                     "using nothing but scrap copper and a single glass bead.",
        "paraphrases": [
            "Using only scrap copper and one glass bead, the fictional inventor Rosalind "
            "Pike built her first clock.",
            "Rosalind Pike's debut clock, an invented device, was made from scrap copper and "
            "a lone glass bead.",
            "With just scrap copper and a single glass bead, Rosalind Pike constructed her "
            "first working clock.",
        ],
    },
    {
        "id": "fact_10",
        "canonical": "The imaginary board game Hollowpoint Chess is played on a board with "
                     "seven concentric rings instead of squares.",
        "paraphrases": [
            "Hollowpoint Chess, a made-up board game, uses seven concentric rings rather than "
            "a square grid.",
            "Instead of squares, the fictional game Hollowpoint Chess is played across seven "
            "concentric rings.",
            "The invented game Hollowpoint Chess replaces the usual square board with seven "
            "concentric rings.",
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
    "The old bridge required minor repairs after several years of steady use.",
    "A new bakery opened on the corner, drawing a modest but curious crowd.",
    "Farmers in the valley reported an average yield for the third year running.",
    "The town council postponed its usual meeting due to a scheduling conflict.",
    "Children gathered at the park most afternoons after school let out.",
    "The harbor saw a modest increase in fishing boats during the calmer months.",
    "A small orchestra rehearsed twice a week in the community hall.",
    "The train timetable shifted slightly to accommodate the new route.",
    "Neighbors often exchanged produce from their gardens during the summer.",
    "The museum's newest exhibit drew a steady stream of weekday visitors.",
    "A light frost settled over the fields on several mornings that month.",
    "The bookstore extended its hours during the final weeks of the term.",
    "Volunteers repainted the community center over a single long weekend.",
    "The road crew finished resurfacing the main avenue ahead of schedule.",
    "A minor power outage affected the eastern district for under an hour.",
    "The choir practiced its winter program in the back room of the chapel.",
    "Local anglers noted an unusually calm week on the lake.",
    "The newspaper ran a short piece on the upcoming harvest fair.",
    "A handful of new benches were installed along the riverside path.",
    "The bakery's morning line grew slightly longer during the colder months.",
    "Students filled the courtyard between lectures on sunny afternoons.",
    "The old clock tower was cleaned and serviced ahead of the anniversary.",
    "A traveling market set up near the square for three days that spring.",
    "The council debated a modest increase to the parking fees downtown.",
    "Several families spent the long weekend camping near the ridge.",
    "The corner cafe added a few new items to its lunch menu.",
    "A quiet crowd gathered to watch the annual boat races from the pier.",
    "The school's garden club planted a new row of vegetables in April.",
    "Repairs to the footbridge took longer than the crew initially expected.",
    "The evening market wound down earlier than usual due to the cold snap.",
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
