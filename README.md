# Duplication and Its Discontents: Memorization vs. Generalization in Fine-tuned Language Models

**Question:** If a fact appears N times in a language model's fine-tuning data, how does that
affect whether the model *memorizes* it verbatim vs. *generalizes* from it — and does heavy
duplication also make the model's outputs less diverse/creative around that content?

This is a small, self-contained research project designed to be finished in ~15-20 days on a
free Colab/Kaggle GPU. It fine-tunes GPT-2 small on a synthetic corpus where you control exactly
how many times each target fact appears, then measures:

1. **Memorization** — can the model reproduce the exact fact string verbatim?
2. **Generalization** — how well does the model handle *paraphrases* of the fact it never saw
   during training (lower perplexity = better generalization)?
3. **Diversity** — how repetitive/generic are the model's generations when prompted about
   heavily-duplicated content vs. rarely-seen content? (distinct-n, embedding diversity)

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
