# Action Plan – ML Foundations in C++

## General Objective

Build a basic, implementation-based introduction to Machine Learning in modern C++, with emphasis on clarity, core model intuition, and simple from-scratch implementations.

This action plan should be understood as a **basic ML fundamentals study**, not as a complete ML foundation and not as sufficient preparation for serious Deep Learning work by itself.

Its role is to:
- build intuition through implementation
- practice C++ in a structured ML project
- create a warm-up foundation before a broader and more serious ML roadmap

---

# Phase Structure

## Sub-Phase 0 – Setup & Project Foundation

**Goal:** Prepare clean C++ ML project

**Time:** 1 day

Checklist:

- [x] Repo structure
- [x] CMake setup
- [x] Docs created
- [x] Build works

---

## Sub-Phase 1 – Math & Data Utilities

**Goal:** Build minimal tools for ML

**Time:** 2–3 days

- [x] Vector ops (add, subtract, scalar multiply)
- [x] Dot product
- [x] Mean
- [x] CSV loading (simple numeric CSV first)
- [x] Train/test split (deterministic first)

---

## Sub-Phase 2 – Linear Regression

**Goal:** Understand regression + GD

**Time:** 3–5 days

- [x] Prediction
- [x] MSE loss
- [x] Gradient
- [x] GD training
- [x] Convergence analysis

---

## Sub-Phase 3 – Optimization (GD / SGD)

**Goal:** Deep understanding of optimization

**Time:** 3–5 days

- [x] Batch GD
- [x] SGD
- [x] Learning rate experiments
- [x] Stability analysis
- [x] Convergence behavior

---

## Sub-Phase 4 – Feature Scaling

**Goal:** Understand preprocessing impact

**Time:** 1–2 days

- [x] Min-max scaling
- [x] Standardization
- [x] Compare results

---

## Sub-Phase 5 – Logistic Regression

**Goal:** Binary classification

**Time:** 4–6 days

- [x] Sigmoid
- [x] Binary prediction interpretation
- [x] Cross-entropy
- [x] Gradient / training loop
- [x] Predictions

---

## Sub-Phase 6 – Evaluation Metrics

**Goal:** Evaluate models properly

**Time:** 1–2 days

- [x] Accuracy
- [x] Precision / Recall
- [x] Confusion matrix

---

## Sub-Phase 7 – k-NN

**Goal:** Non-parametric model

**Time:** 2–3 days

- [x] Distance
- [x] k selection
- [x] Predictions

---

## Sub-Phase 8 – Decision Tree (Simplified)

**Goal:** Rule-based learning

**Time:** 4–6 days

- [x] Splits
- [x] Impurity
- [x] Tree structure

---

## Sub-Phase 9 – Wrap-Up

**Goal:** Close the repo cleanly and position it correctly as a basic ML fundamentals study

**Time:** 0.5–1 day

- [x] Update `README.md` disclaimer and project positioning
- [x] Update `docs/general/action-plan.md`
- [x] Update `docs/general/ml-foundations.md`
- [x] Refactor `main.cpp` to keep it understandable
- [x] Add short wrap-up summary

---

## Contents

1. Linear Regression
2. Gradient Descent / SGD
3. Logistic Regression
4. Feature Scaling
5. Evaluation Metrics
6. k-NN
7. Decision Tree (simplified)

---

## Final Positioning of This Action Plan

This action plan should be treated as:
- a completed basic ML implementation study in C++
- an introductory classical ML warm-up
- a structured project for learning core concepts through code

It should **not** be treated as:
- a complete ML curriculum
- a complete bridge to Deep Learning
- the final theory foundation required for serious modern ML work

The next step after this repository is a new and more serious ML roadmap designed to cover the missing theoretical, statistical, optimization, and modeling foundations before moving into Deep Learning.