---
title: "On the Comparative Performance of Machine Learning and Economic Models for Risky Decision-Making"
author: "Mateus Hiro Nagata"
date: "6/1/2025"
output: github_document
---


---

# 🧠 On the Comparative Performance of Machine Learning and Economic Models for Risky Decision-Making

This repository contains the **reproduction package** for the working paper *"On the Comparative Performance of Machine Learning and Economic Models for Risky Decision-Making"*.

The project investigates how machine learning methods can model and predict human decision-making under risk, comparing traditional economic frameworks (like **Cumulative Prospect Theory**) with **data-driven** and **hybrid** approaches.

---

## 📘 Overview

The goal is to compare three broad classes of models:

1. **Economic Models** – grounded in behavioral economics (e.g., Cumulative Prospect Theory).
2. **Pure Machine Learning Models** – fully data-driven (e.g., random forests, regularized regressions).
3. **Hybrid Models** – neural networks that embed economic theory into their structure or loss function.

This comparison provides insights into how well each class captures **utility curvature** and **probability weighting**, two central features of decision-theoretic behavior.

---

## 📁 Repository Structure

| Script                             | Description                                                                                                                                                   |
| ---------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `DT_ML_Data.py`                    | Prepares and preprocesses the dataset for training and evaluation. Includes feature construction and transformations related to decision-theoretic variables. |
| `DT-ML MLE.py`                     | Implements Maximum Likelihood Estimation (MLE) for parameter estimation in decision-theoretic models.                                                         |
| `DT-ML NN.py`                      | Defines and trains hybrid neural network models incorporating economic theory (e.g., constrained layers or custom loss terms).                                |
| `DT-ML Regularized Regressions.py` | Trains regularized regression models (LASSO, Ridge, Elastic Net) as baseline comparators.                                                                     |
| `DT-ML Trees.py`                   | Implements tree-based ML models such as Random Forests and Gradient Boosted Trees.                                                                            |

---

## ⚙️ Requirements

This project is written in **Python 3.14.0**.
Core dependencies include:

```bash
numpy
pandas
scikit-learn
tensorflow  # or pytorch
matplotlib
seaborn
scipy
```

To install dependencies:

```bash
pip install -r requirements.txt
```
---

## 🚀 Usage

To reproduce results from the paper:

1. **Prepare the data**

   ```bash
   python DT_ML_Data.py
   ```

2. **Train and evaluate models**

   ```bash
   python DT-ML_Trees.py
   python DT-ML_Regularized Regressions.py
   python DT-ML_NN.py
   ```

3. **Analyze outputs**

   Model metrics, parameter estimates, and figures are saved in the designated results folders or printed to the console.

---

## 📊 Outputs

The scripts generate:

* Comparative performance metrics (accuracy, RMSE, log-likelihood)
* Visualizations of **utility** and **probability weighting functions**
* Tables comparing **economic**, **ML**, and **hybrid** models

---

## 🧩 Research Context

This work contributes to bridging **behavioral economics** and **machine learning**, by:

* Testing empirical validity of decision-theoretic assumptions
* Using ML to discover flexible, nonparametric patterns in choice behavior
* Building interpretable hybrid models combining structure and prediction power

---

## Data
- The data is from Vieider, Ferdinand M., et al. "Common components of risk and uncertainty attitudes across contexts and domains: Evidence from 30 countries." Journal of the European Economic Association 13.3 (2015): 421-452
- Unfortunately, I do not have authorization to publish the data.

--- 

## 📄 Citation

If you use this code or reproduce the results, please cite:

> Mateus Hiro Nagata. *On the Comparative Performance of Machine Learning and Economic Models for Risky Decision-Making.* Working Paper, HEC Paris, 2025.

---

## 🪪 License

This project is released under the [MIT License](LICENSE).

---


## Project Structure

```
├── data/
├── code/
│   ├── DT_ML_Data.py #Cleaning data
│   ├── DT-ML_MLE.py  # MLE estimation of CPT
│   ├── DT-ML_NN.py   # NN predictions and hybrid models 
│   ├── DT-ML_Regularized_Regression.py # LASSO
│   ├── DT-ML_Trees.py # Random Forests and other variations
├── output/
│   ├── figures/       # Generated plots
│   └── tables/        # Results tables
└── README.md
```


