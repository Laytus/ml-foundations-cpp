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
# ML Foundations in C++

## Purpose

Study basic Machine Learning concepts by implementing simplified models from scratch in C++, without ML/DL libraries.

This document should be understood as describing a **basic ML fundamentals study**, not a complete Machine Learning foundation and not a sufficient bridge to Deep Learning on its own.

The purpose of this repository is to:
- build intuition for core ML ideas through implementation
- practice C++ in a structured multi-file ML project
- create a warm-up foundation before a broader and more serious ML roadmap

Focus on:

- mathematical intuition
- clean implementation
- reproducibility
- simplicity over abstraction

---

## Main Goal

Build a basic ML implementation foundation in C++ through the study of simple classical models and core concepts, with special emphasis on:

- optimization intuition
- gradient descent
- SGD
- loss functions
- training loops
- numerical reasoning
- simple data handling
- basic evaluation

This goal is intentionally limited.

It is meant to provide:
- introductory model intuition
- from-scratch implementation practice
- a clean study codebase

It is **not** meant to provide:
- a complete ML theory foundation
- a full statistical learning background
- full readiness for serious Deep Learning work by itself

---

## Core decisions

1. **Linear Regression**
2. **Logistic Regression**
3. **Gradient Descent (basic to intermediate understanding)**
4. **k-Nearest Neighbors (k-NN)**
5. **Basic Feature Scaling**
6. **Decision Tree (simplified version)**

These were chosen because they are useful for:
- understanding fundamental ML mechanics
- learning multiple model families
- seeing how predictions, losses, and simple training behavior work in code

They should be viewed as a **first implementation-oriented set of models**, not as a complete map of modern ML.

---

## Priority of Concepts

### Optimization (Critical Focus)

The most important concept in this repository is:

> **Gradient Descent and Stochastic Gradient Descent (SGD)**

Focus areas:
- cost functions
- gradients
- parameter updates
- learning rate
- convergence behavior
- batch vs stochastic updates
- debugging training loops

This is important because optimization is one of the key conceptual bridges between classical ML and later Deep Learning.

At the same time, the current repository covers this at a **basic foundational level**, not at the full level required for advanced ML or DL training systems.

---

## Real-World Relevance

This phase is loosely aligned with real ML usage in areas such as:

- regression → pricing, forecasting
- classification → fraud detection, signals
- preprocessing → important in tabular problems
- tree-based reasoning → introduced in simplified form
- optimization → central to trainable models

However, this repository should not be interpreted as a full industry-preparation ML curriculum.

It is better understood as:
- a conceptual warm-up
- an implementation study
- an early intuition-building phase

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
- full statistical learning theory
- advanced model selection and hyperparameter search
- modern production ML systems

These topics belong to a later and more serious ML phase.

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
- understanding

Avoid:

- overengineering
- unnecessary abstraction

### 4. Mathematical Understanding

Every model should be understood both:

- mathematically
- programmatically

### 5. Scope awareness

This project intentionally favors:
- simplified implementations
- small understandable modules
- conceptual clarity over full ML completeness

That is a deliberate design choice, not an accident.

---

## Output / Visualization Strategy

- Save results to CSV / JSON when useful
- Inspect results externally when useful
- Avoid building heavy UI systems

The priority is:

- ML intuition
- clean implementation
- study value of the codebase

---

## Expected Final Outcome

By the end of this repository's intended scope:

- basic classical ML models are implemented in C++
- core introductory optimization behavior is more intuitive
- the project serves as a clean ML/C++ study reference
- the repo acts as a useful warm-up before a more complete ML roadmap

This should **not** be interpreted as meaning:
- full ML mastery
- full theoretical readiness for Deep Learning
- completion of all important ML foundations

Instead, this repository should be positioned as:
- a completed basic ML fundamentals study
- an implementation-driven introduction
- a precursor to a broader and more serious ML learning plan