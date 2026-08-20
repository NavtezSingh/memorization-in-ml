# Duplication and Its Discontents: Memorization vs. Generalization in Fine-tuned Language Models

**Question:** If a fact appears N times in a language model's fine-tuning data, how does that
affect whether the model *memorizes* it verbatim vs. *generalizes* from it — and does heavy
duplication also make the model's outputs less diverse/creative around that content?

This is a small, self-contained research project designed to be finished in ~15-20 days on a
free Colab/Kaggle GPU. It fine-tunes GPT-2 small on a synthetic corpus where you control exactly
how many times each target fact appears, then measures:

1. **Memorization** — a continuous score (0-1): given only the first ~30% of a fact sentence,
   how much of the true continuation does the model reproduce, word-for-word, in order?
2. **Generalization** — how well does the model handle *paraphrases* of the fact it never saw
   during training (lower perplexity = better generalization)?
3. **Diversity** — how repetitive/generic are the model's generations when prompted about
   heavily-duplicated content vs. rarely-seen content? (distinct-n, embedding diversity)

**Important methodology notes (learned the hard way, through two failed runs):**

*Run 1* hit a ceiling effect — memorization saturated at 1.0 and perplexity was flat at every
duplication level, because a too-small corpus (4 facts, 10 filler sentences) was trivially
overfit within 3 epochs regardless of duplication count. Fixed by expanding to 10 facts and 40
filler sentences.

*Run 2* tried to control for this by fixing total training steps (`--max_steps`) instead of
epochs, reasoning that a bigger corpus (higher duplication) shouldn't get more total gradient
updates just because it's bigger. This backfired badly: fixing total steps across
differently-sized corpora means the *smallest* corpus (least duplication) ends up training for
far more epochs than the *largest* one — e.g. at `max_steps=200`, the least-duplicated corpus
trained for ~50 epochs while the most-duplicated one trained for under 1. The low-duplication
model catastrophically overfit and collapsed (perplexity exploded from ~50 to ~6000), and all
conditions converged to a similarly broken state — flat again, just flat somewhere much worse.

**The correct design is `--epochs 1` (now the default in `train.py`).** With exactly one pass
over the corpus, each occurrence of a fact is touched by the optimizer exactly once — so the
total number of times the model sees a given fact string during training equals its
duplication count *exactly*. No epoch confound, no steps confound. This is simpler than either
previous attempt and is the right way to isolate the effect of duplication count.

Also fixed along the way:
- **10 target facts** (not 4) instead of a handful, for finer-grained averages
- **40 unique filler sentences** (not 10) so the surrounding "world" isn't trivial to overfit
- **Continuous, position-sensitive memorization score** instead of a binary
  memorized/not-memorized threshold, so results have real resolution instead of collapsing to
  a handful of discrete values
- **Lower learning rate (2e-5, was 5e-5)** to slow down how fast the model overfits

## Why this project

- Ties directly to data-centric interpretability: how training data composition shapes
  downstream model behavior. Duplicated text is a well-documented driver of memorization in
  real LLMs — this project reproduces that phenomenon at small scale, in a fully controlled way.
- Extends naturally into AI creativity: the diversity analysis asks whether memorization comes
  at the cost of creative/varied output.
- Cheap to run: GPT-2 small (124M params) fine-tunes in minutes per run on a free Colab GPU.

## Project structure

```
dup-project/
├── README.md
├── requirements.txt
├── src/
│   ├── data_gen.py              # builds the synthetic corpus with controlled fact duplication
│   ├── train.py                 # fine-tunes GPT-2 small for each duplication level
│   ├── evaluate_memorization.py # exact-match recall + perplexity on paraphrases
│   └── evaluate_diversity.py    # distinct-n + embedding diversity of generations
├── notebooks/
│   └── colab_quickstart.ipynb   # runs the whole pipeline end-to-end in Colab
├── data/                        # generated corpora land here
└── outputs/                     # checkpoints, metrics, plots land here
```

## Day-by-day plan (~15-20 days)

| Days | Task |
|---|---|
| 1-3 | Run `data_gen.py`, sanity-check the corpus, tune the set of target facts and duplication levels (e.g. 1x, 5x, 20x, 100x) |
| 4-8 | Run `train.py` for each duplication level, save checkpoints |
| 9-13 | Run `evaluate_memorization.py`: plot exact-match rate and paraphrase perplexity vs. duplication count |
| 14-16 | Run `evaluate_diversity.py`: plot distinct-n / embedding diversity vs. duplication count |
| 17-20 | Write up findings (1-2 pages), clean up repo, push to GitHub with plots in README |

## Quickstart

Open `notebooks/colab_quickstart.ipynb` in Google Colab (Runtime → GPU), run all cells. It
installs dependencies, generates data, fine-tunes small models, and produces the two evaluation
plots described above. Each script can also be run standalone — see comments at the top of each
file in `src/`.

## What "done" looks like

A repo with:
- The pipeline above, working end-to-end
- Two plots: (1) memorization/generalization vs. duplication count, (2) diversity vs.
  duplication count
- A short write-up interpreting the results — e.g. "memorization rate rises sharply past ~20
  duplications, while generalization to paraphrases plateaus/declines, and generation diversity
  around heavily duplicated facts drops by X%"

That's a complete, presentable undergraduate research artifact — exactly the kind of thing worth
linking in a PhD/internship application.
