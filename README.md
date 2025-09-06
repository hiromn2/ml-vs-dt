---
title: "On the Comparative Performance of Machine Learning and Economic Models for Risky Decision-Making"
author: "Mateus Hiro Nagata"
date: "6/1/2025"
output: github_document
---

# ML vs DT: On the Comparative Performance of Machine Learning and Economic Models for Risky Decision-Making 

## Overview

This repository contains the reproduction package for my working paper on machine learning modelling of utility and probability weighting in decision theory.

## Project Structure

```
├── data/
│   ├── raw/           # Original experimental data
│   └── processed/     # Cleaned datasets
├── code/
│   ├── 01_data_cleaning.R
│   ├── 02_analysis.R
│   └── 03_figures.R
├── output/
│   ├── figures/       # Generated plots
│   └── tables/        # Results tables
└── README.md
```

## Requirements

### Software


### Data
- [Describe your data source - e.g., "Experimental data from classroom sessions" or "Simulated game data"]
- [Note any privacy considerations or data access restrictions]

## Running the Analysis

To reproduce all results, run the scripts in order:

```{r reproduction, eval=FALSE}
# Clean and prepare data
source("code/01_data_cleaning.R")

# Run main analysis
source("code/02_analysis.R") 

# Generate figures and tables
source("code/03_figures.R")
```

**Expected runtime:** Approximately [X minutes] on a standard laptop.

## Main Results

The analysis produces:
- **Figure 1:** [Brief description - e.g., "Learning curves by treatment group"]
- **Figure 2:** [Brief description - e.g., "Strategy convergence over rounds"]
- **Table 1:** [Brief description - e.g., "Summary statistics by condition"]
- **Table 2:** [Brief description - e.g., "Regression results for learning rates"]

## Key Findings

[1-2 sentences summarizing your main results]
