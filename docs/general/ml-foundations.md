# ML Foundations in C++

## Purpose

Learn core Machine Learning concepts by implementing models from scratch in C++, without ML/DL libraries.

Focus on:

- mathematical understanding
- clean implementation
- reproductibility
- simplicity over abstraction

---

## Main Goal

Build a strong ML foundation in C++ through the implementation of core classical models and concepts, with special emphasis on:

- optimization
- gradient descent
- SGD
- loss functions
- training loops
- numerical reasoning
- data handling
- evaluation

---

## Core decisions

1. **Linear Regression**
2. **Logistic Regression**
3. **Gradient Descent (deep understanding)**
4. **k-Nearest Neighbors (k-NN)**
5. **Basic Feature Scaling**
6. **Decision Tree (simplified version)**

These are chosen because they represent the **core techniques most used in real-world ML**, especially in:

- finance
- tabular data problems
- applied ML systems

---


## Priority of Concepts

### Optimization (Critical Focus)

The most important concept in this phase is:

> **Gradient Descent and Stochastic Gradient Descent (SGD)**

Focus areas:
- cost functions
- gradients
- parameter updates
- learning rate
- convergence behavior
- batch vs stochastic updates
- debugging training loops

This is fundamental not only for Deep Learning, but for **Machine Learning as a whole**.

---

## Real-World Relevance

This phase is aligned with how ML is actually used in practice:

- regression → pricing, forecasting
- classification → fraud detection, signals
- feature engineering → critical in finance
- tree-based reasoning → simplified via decisions trees
- optimization → core of all trainable models

---

## Non-Goals of this phase

To maintain focus, the following are out of scope:

- advanced ensemble methods (XGBoost, LightGBM)
- SVM
- PCA
- clustering
- reinforcement learning
- deep learning architectures
- ML frameworks (scikit-learn, PyTorch, TensorFlow)

---

## Implementation Philosophy

### 1. C++ first

All core implementations must be in C++.

### 2. From scratch

No ML libraries will be used.

Allowed:

- STL
- minimal math/utilities if needed
- CSV export
- simple helper utilities

### 3. Clarity over Complexity

Prioritize:

- readability
- correctness
- understading

Avoid:

- overengineering
- unnecessary abstraction

### 4. Mathematical Understanding

Every model must be understood both:

- mathematically
- programmatically

---

## Output / Visualization Strategy

- Save results to CSV / JSON
- Inspect results externally when useful
- AVoid building heavy UI systems

The priority is:

- ML understangin
- clean implementation

---

## Expected Final Outcome

By the end of this phase:

- strong understanding of classical ML models
- ability to implement them from scratch in C++
- solid intuition for optimization and training
- readiness to apply ML to real-world problems