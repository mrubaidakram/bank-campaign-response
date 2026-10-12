# Bank Campaign Response

> A leakage-aware tabular classification workflow for predicting term-deposit subscription from public campaign records.

## Goal

Chronological train/validation/test evaluation with a regularized logistic-regression pipeline. Excludes call duration by default because it is only known after the call.

## Dataset

UCI Bank Marketing, including the 41,188-row `bank-additional-full.csv`. The official archive is downloaded and extracted at run time; data is excluded from Git.

**Official source:** [https://archive.ics.uci.edu/dataset/222/bank+marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing)
**Data terms:** UCI lists CC BY 4.0. Cite Moro, Rita, and Cortez (2014) and retain attribution when adapting or sharing results.

Raw dataset files are not committed. First run downloads data into the ignored `data/` directory.

## Run

Python 3.10 or newer is required. Create a virtual environment, install the local package, and train:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .
python -m bank_campaign.train --data-dir data --output-dir artifacts
```

The first run may download the dataset. Checkpoint and metrics output is written to `artifacts/`.

## Reproducibility and evaluation

- Training, validation, and test responsibilities are documented in the training code.
- Random seeds and key hyperparameters are explicit CLI options or constants.
- Test metrics are reported only after model selection on validation data.
- Results are not published in this README until a reproducible run has produced them.

## Stack

Machine learning · tabular data · leakage prevention · scikit-learn

## Scope

This is an independently built portfolio project using public benchmark data. It contains no client records, employer code, or claims that it reproduces private production systems. Dataset attribution and license obligations remain with users of the data.

## Validation status

Python source syntax has been checked. Dependencies have not been installed and training has not been run; runtime behavior and model quality are not yet verified.
