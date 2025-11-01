---
title: "On the Comparative Performance of Machine Learning and Economic Models for Risky Decision-Making"
author: "Mateus Hiro Nagata"
date: "6/1/2025"
output: github_document
---

# On the Comparative Performance of Machine Learning and Economic Models for Risky Decision-Making 

## Overview

This repository contains the reproduction package for my working paper on machine learning modelling of utility and probability weighting in decision theory. The fundamental idea is to compare economic models (such as cumulative prospect theory), pure ML models (random forests, for example) and hybrid models - economic theory assisted neural networks.

## Project Structure

```
├── data/
│   ├── raw/           # Original experimental data
│   └── processed/     # Cleaned datasets
├── code/
│   ├── DT_ML_Data.py #Cleaning data
│   ├── DT-ML_MLE.py  # MLE estimation of CPT
│   ├── DT-ML_NN.py   # NN predictions and hybrid models 
│   ├── DT-ML_Regularized_Regression.py # LASSO, ->  not insightful
│   ├── DT-ML_Trees.py # Random Forests and other variations
├── output/
│   ├── figures/       # Generated plots
│   └── tables/        # Results tables
└── README.md
```


### Software

Python 3.14.0

### Data
- The data is from Vieider, Ferdinand M., et al. "Common components of risk and uncertainty attitudes across contexts and domains: Evidence from 30 countries." Journal of the European Economic Association 13.3 (2015): 421-452
- Unfortunately, I do not have authorization to publish the data.


## Key Findings

What constitutes the pinnacle of decision under risk models? This paper addresses
this question by comparing a range of models, from popular economic models to black-box
machine learning algorithms, as well as hybrid approaches in predicting the choice of certainty
equivalents of risky prospects. The findings demonstrate that there is a relevant gain in descriptive
prowess in using machine learning techniques. However, this indicates heterogeneity in the
population rather than inadequacy of the economic models.
