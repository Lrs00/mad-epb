# MAD and Empirical Partially Bayes Testing

This repository contains simulation code for comparing variance-based and mean-absolute-deviation-based procedures in large-scale multiple testing.

The main goal is to study whether the choice of within-feature scale statistic continues to affect inference when information about variability is borrowed across testing problems.

## Simulation settings

We consider four error distributions:

- Gaussian
- Mixture Normal
- Mixture Laplace
- Scaled Uniform

Unless otherwise specified, the simulations use:

- `G = 1000` hypotheses
- `R = 100` simulation replicates
- `10%` non-null hypotheses
- `β_signal = 4`
- `n = 4, 5, 6, 8, 10, 20, 30, 40`
- Benjamini-Hochberg FDR level `0.05`

For the mixture settings, different contamination proportions are considered, including

- `epsilon = 0.05`
- `epsilon = 0.10`
- `epsilon = 0.15`

## Methods

The simulation study compares several procedures, including:

- Ordinary t test
- MAD-normalized test
- LIMMA
- SD-based nonparametric empirical Bayes
- MAD-based parametric empirical partially Bayes 
- MAD-based nonparametric empirical partially Bayes 

Both parametric and nonparametric versions are explored for some empirical Bayes procedures.

## Performance measures

The methods are compared using quantities such as:

- Power
- False discovery rate (FDR)
- True positives
- False positives
- Top-100 true positives
- Precision and recall among top-ranked hypotheses
- Average precision

## Repository structure

- `src/`: reusable functions
- `scripts/`: simulation scripts for each noise setting
- `notebooks/`: exploratory and development notebooks
- `results/`: saved simulation summaries
- `figures/`: generated figures

## Status

This repository accompanies an ongoing research project and is currently under active development.
