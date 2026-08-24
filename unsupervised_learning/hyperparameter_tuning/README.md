# Hyperparameter Tuning & Bayesian Optimization

## 📌 Project Overview

This project focuses on hyperparameter optimization techniques for Machine Learning models, moving beyond traditional exhaustive search methods to probabilistic model-based approaches. 

Through this project, I explore:
* The fundamentals of **Gaussian Processes (GP)** and **Kriging** as surrogate models for unknown target functions.
* The application of **Bayesian Optimization** to find optimal hyperparameter configurations efficiently with minimal function evaluations.
* The mechanics of **Acquisition Functions** (such as Expected Improvement, Knowledge Gradient, and Entropy Search) to navigate the trade-off between exploration and exploitation.
* Hands-on implementation and experimentation using Python libraries `GPy` and `GPyOpt`.

---

## 🚀 Key Topics & Concepts Covered

* **Hyperparameter Tuning Methods:** Comparing Grid Search, Random Search, and Sequential Model-Based Optimization (SMBO).
* **Gaussian Process Regression:** Modeling mean functions, covariance kernels, and uncertainty bounds.
* **Bayesian Optimization Loop:** Updating posterior distributions and evaluating acquisition functions to sample optimal points sequentially.
* **Tooling:** Implementing GP models and optimization tasks with `GPy` and `GPyOpt`.

---

## 🛠️ Environment & Prerequisites

* **OS:** Ubuntu 20.04 LTS
* **Python Version:** 3.9
* **Main Libraries:**
  * `numpy` (v1.25.2)
  * `GPy`
  * `GPyOpt`
* **Code Style:** `pycodestyle` (v2.11.1)

---

## 📂 Project Structure

```text
.
├── README.md               # Project documentation
├── 0-bayes_opt.py          # Bayesian Optimization tasks and implementations
└── ...