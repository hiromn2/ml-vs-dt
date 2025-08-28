---
title: "Learning in Games: Reproduction Package"
author: "Mateus Hiro Nagata"
date: "6/1/2025"
output: github_document
---

```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE)
```

# Learning in Games: [Your Paper Title Here]

## Overview

This repository contains the reproduction package for my working paper on learning in games. The paper analyzes [brief description of your research question - e.g., "how players adapt their strategies over repeated interactions" or "convergence properties in experimental game settings"].

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
- R version 4.0 or higher
- Required packages:
```{r packages, eval=FALSE}
install.packages(c("tidyverse", "ggplot2", "stargazer"))
# Add other packages you use
```

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

## Contact

For questions about this reproduction package, contact: [your email]

---

*This package was created for [Course Name/Assignment] at [University Name].*
# ml-vs-dt
