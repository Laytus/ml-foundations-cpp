# Logistic Regression

## Purpose

The goal of this sub-phase is to build the first binary classification model in the project.

At this stage of the project, Logistic Regression is studied as the classification counterpart of Linear Regression:
- Linear Regression predicts continuous values
- Logistic Regression predicts probabilities for binary classes

This sub-phase is important because it introduces:
- classification
- probability-like model outputs
- a new loss function
- threshold-based predictions

So Logistic Regression is the natural next step after Linear Regression, optimization, and feature scaling.

---

## Why Logistic Regression matters

So far, the project has focused on regression problems, where the target is a continuous numeric value.

But many Machine Learning problems are classification problems.

For example:
- spam vs not spam
- accepted vs rejected
- default vs non-default
- positive vs negative class

In these problems, the goal is not to predict an arbitrary real number. The goal is to predict which class an input belongs to.

Logistic Regression is one of the simplest and most important models for binary classification.

It is important because it introduces the first full classification pipeline in the project while still remaining mathematically simple.

---

## Binary classification

In the first version of Logistic Regression in this project, the task is binary classification.

That means the target can take only two possible values:

$$
y \in \{0, 1\}
$$

Where:
- `0` represents one class
- `1` represents the other class

So the model should learn how to map an input value to a prediction that helps separate these two classes.

---

## Logistic Regression as a model

Logistic Regression begins with the same linear part used in Linear Regression:

$$
z = wx + b
$$

This linear expression is often called the logit or score.

But unlike Linear Regression, Logistic Regression does not use this raw linear value directly as the final output.

Instead, it transforms that value through the sigmoid function.

So the model becomes:

$$
\hat{y} = \sigma(wx + b)
$$

where:
- \(z = wx + b\)
- \(\sigma(z)\) is the sigmoid function

This means Logistic Regression is still a linear model in its structure, but it produces a bounded nonlinear output.

---

## Sigmoid

### Purpose

The sigmoid function is the first new mathematical building block of Logistic Regression.

Its role is to transform any real-valued input into a bounded output between `0` and `1`.

This is exactly what Logistic Regression needs, because binary classification requires outputs that can be interpreted as probabilities or probability-like scores.

### Definition

The sigmoid function is:

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

where:
- \(z\) is any real number

The output of the sigmoid function always satisfies:

$$
0 < \sigma(z) < 1
$$

So no matter how large or how small the input is, the result stays strictly between `0` and `1`.

### Where sigmoid comes from

The sigmoid function does not appear in Logistic Regression as an arbitrary squashing function.

Its mathematical origin comes from modeling the **log-odds** of a binary event as a linear function.

Suppose the model wants to represent the probability that:

$$
y = 1
$$

for a given input.

Call that probability:

$$
p = P(y=1 \mid x)
$$

Because this is a probability, it must satisfy:

$$
0 \le p \le 1
$$

A simple linear expression such as:

$$
wx + b
$$

cannot be used directly as a probability because it can take any real value.

So Logistic Regression does not model the probability directly with a linear expression.

Instead, it models the **log-odds**.

#### Step 1 — Define the odds

If the probability of class `1` is \(p\), then the probability of class `0` is:

$$
1-p
$$

So the odds of class `1` are:

$$
\frac{p}{1-p}
$$

This compares how likely class `1` is relative to class `0`.

#### Step 2 — Take the log-odds

Now take the logarithm of the odds:

$$
\log\left(\frac{p}{1-p}\right)
$$

This quantity is called the **logit**.

The important reason for doing this is that:
- probabilities live in \((0,1)\)
- odds live in \((0,\infty)\)
- log-odds live in \(( -\infty, +\infty )\)

That makes log-odds compatible with a linear model, because a linear expression can also take any real value.

So Logistic Regression makes the modeling assumption:

$$
\log\left(\frac{p}{1-p}\right) = wx + b
$$

If we define:

$$
z = wx + b
$$

then this becomes:

$$
\log\left(\frac{p}{1-p}\right) = z
$$

#### Step 3 — Solve for the probability

Now solve for \(p\).

Start from:

$$
\log\left(\frac{p}{1-p}\right) = z
$$

Exponentiate both sides:

$$
\frac{p}{1-p} = e^z
$$

Multiply both sides by \((1-p)\):

$$
p = e^z(1-p)
$$

Expand:

$$
p = e^z - e^z p
$$

Group the \(p\) terms together:

$$
p + e^z p = e^z
$$

Factor out \(p\):

$$
p(1 + e^z) = e^z
$$

Divide both sides:

$$
p = \frac{e^z}{1 + e^z}
$$

Now divide numerator and denominator by \(e^z\):

$$
p = \frac{1}{1 + e^{-z}}
$$

So the probability becomes:

$$
p = \sigma(z) = \frac{1}{1 + e^{-z}}
$$

This is exactly the sigmoid function.

### What this means

So in Logistic Regression, the sigmoid comes from the assumption that the **log-odds** of class `1` are linear in the input:

$$
\log\left(\frac{p}{1-p}\right) = wx + b
$$

and then solving that equation for \(p\).

This gives Logistic Regression a very important interpretation:
- the linear part \(wx+b\) models the log-odds
- the sigmoid converts those log-odds into a probability-like output in \((0,1)\)

That is the real mathematical origin of the sigmoid in Logistic Regression.

### Why sigmoid is useful

The linear expression:

$$
z = wx + b
$$

can take any real value:
- very negative
- zero
- very positive

That is not suitable directly for binary classification probabilities.

The sigmoid solves this by compressing the real line into the interval:

$$
(0, 1)
$$

So sigmoid turns an unrestricted score into a probability-like output.

### Intuition

The sigmoid behaves in three main regions.

#### Very negative input

If \(z\) is a large negative number, then:

$$
\sigma(z) \approx 0
$$

So the model strongly favors class `0`.

#### Input near zero

If \(z = 0\), then:

$$
\sigma(0) = \frac{1}{2} = 0.5
$$

So the model is maximally uncertain between the two classes.

#### Very positive input

If \(z\) is a large positive number, then:

$$
\sigma(z) \approx 1
$$

So the model strongly favors class `1`.

### Example values

A few useful reference values are:

#### At \(z = 0\)

$$
\sigma(0) = 0.5
$$

#### At \(z = 2\)

$$
\sigma(2) = \frac{1}{1 + e^{-2}} \approx 0.8808
$$

#### At \(z = -2\)

$$
\sigma(-2) = \frac{1}{1 + e^{2}} \approx 0.1192
$$

These examples show how the sigmoid smoothly maps negative values toward `0` and positive values toward `1`.

### Important properties

#### Bounded output

The output is always between `0` and `1`.

#### Monotonicity

If:

$$
z_1 < z_2
$$

then:

$$
\sigma(z_1) < \sigma(z_2)
$$

So larger scores always produce larger predicted probabilities.

#### Smoothness

The sigmoid is smooth and differentiable everywhere.

This is essential for gradient-based optimization.

### Why sigmoid matters for this project

In the current project, sigmoid is important because it creates the first bridge between:
- linear model scores
- bounded prediction outputs
- binary classification
- probability-like interpretation

Without sigmoid, the model would still behave like a regression model.

With sigmoid, the model becomes suitable for classification.

### First scope in this project

For the first version, sigmoid should stay simple:
- scalar input
- scalar output
- direct mathematical implementation
- clean numeric tests

That is enough for a strong first implementation.

### Important practical note

Although the sigmoid output can be interpreted like a probability, the project should treat it first as a probability-like model output.

The deeper probabilistic interpretation becomes more meaningful once cross-entropy and prediction thresholds are introduced.

So the first role of sigmoid is simple:
- map scores into `(0, 1)`
- prepare the model for binary classification

---

## Binary prediction interpretation

### Purpose

Once Logistic Regression produces a sigmoid output, the next question is how that output should be interpreted.

The sigmoid does not directly produce a class label such as `0` or `1`.

Instead, it produces a probability-like score in the interval:

$$
(0, 1)
$$

So this concept explains how Logistic Regression moves from:
- a linear score
- to a sigmoid output
- to a binary class decision

### Two levels of interpretation

For the first version of the project, the Logistic Regression output should be interpreted at two levels.

#### 1. Probability-like output

The model first computes:

$$
\hat{y} = \sigma(wx + b)
$$

This value is not yet the final class prediction.

Instead, it is a probability-like score expressing how strongly the model favors class `1` over class `0`.

So:
- values near `0` favor class `0`
- values near `1` favor class `1`
- values near `0.5` indicate uncertainty

#### 2. Binary class prediction

To convert the sigmoid output into an actual class label, the model must apply a decision threshold.

This produces the binary prediction:

$$
\hat{c} \in \{0, 1\}
$$

So binary prediction requires one extra step after sigmoid.

### Why a threshold is needed

A value such as:

$$
0.83
$$

is meaningful as a probability-like score, but the model still needs a rule to decide whether that should count as class `1` or class `0`.

Without a threshold, the model output remains continuous.

So Logistic Regression becomes a true classifier only once a thresholding rule is introduced.

### Standard threshold for the first version

For the first version of this project, the prediction rule should be:

$$
\hat{c} =
\begin{cases}
1 & \text{if } \hat{y} \ge 0.5 \\
0 & \text{if } \hat{y} < 0.5
\end{cases}
$$

This is the standard first choice because:
- sigmoid outputs are centered around `0.5`
- \(\sigma(0) = 0.5\)
- the model is maximally uncertain at that point

So `0.5` is the natural first decision boundary.

### Interpretation of common output regions

#### Output near 0

If:

$$
\hat{y} \approx 0
$$

then the model strongly favors class `0`.

#### Output near 0.5

If:

$$
\hat{y} \approx 0.5
$$

then the model is uncertain between the two classes.

This corresponds to a score near:

$$
wx + b \approx 0
$$

#### Output near 1

If:

$$
\hat{y} \approx 1
$$

then the model strongly favors class `1`.

### Relationship with the linear score

Because:

$$
\hat{y} = \sigma(wx+b)
$$

and sigmoid is monotonic, thresholding at `0.5` is equivalent to thresholding the linear score at `0`.

That is:

$$
\sigma(z) \ge 0.5 \iff z \ge 0
$$

So for the first version of Logistic Regression:
- if \(wx+b \ge 0\), predict class `1`
- if \(wx+b < 0\), predict class `0`

This is a very important interpretation, because it shows that Logistic Regression still uses a linear decision boundary in the input space.

### Why this matters for the project

In the current project, binary prediction interpretation is important because it creates the bridge between:
- probability-like outputs
- classification decisions
- later evaluation metrics

Without this step, the model can produce sigmoid values, but it still cannot behave as a full classifier.

With thresholding, the model becomes able to produce actual class predictions.

### First scope in this project

For the first version, binary prediction interpretation should stay simple:
- use sigmoid output as the probability-like value
- use a fixed threshold of `0.5`
- produce class predictions in `{0, 1}`

That is enough for a strong first classification workflow.

### Important practical note

Later in Machine Learning, thresholds do not always need to stay fixed at `0.5`.

But for the first version of this project, `0.5` is the correct choice because it is the cleanest and most standard interpretation of Logistic Regression output.

---

## Cross-entropy loss

### Purpose

Once Logistic Regression produces probability-like outputs, the next question is how to measure whether those outputs are good or bad.

For regression, the project used MSE.

But for binary classification with sigmoid outputs, the natural first loss is cross-entropy.

So cross-entropy provides the training objective that connects:
- predicted probabilities
- true binary labels
- gradient-based learning

### Why MSE is not the main loss here

The sigmoid output:

$$
\hat{y} = \sigma(wx+b)
$$

is a probability-like value in:

$$
(0,1)
$$

and the true target is:

$$
y \in \{0,1\}
$$

Although it is mathematically possible to compare these using MSE, Logistic Regression is usually trained with cross-entropy because it is better aligned with binary probability modeling.

So in the first version of Logistic Regression in this project, cross-entropy should be treated as the main loss.

### Definition for one sample

For one sample, the binary cross-entropy loss is:

$$
L(y, \hat{y}) = -\left[y\log(\hat{y}) + (1-y)\log(1-\hat{y})\right]
$$

where:
- \(y\) is the true label in \(\{0,1\}\)
- \(\hat{y}\) is the predicted probability-like output in \((0,1)\)

This is the basic loss used in Logistic Regression.

### Interpretation by class

Because \(y\) can only be `0` or `1`, the formula simplifies in two important cases.

#### Case 1 — true class is 1

If:

$$
y = 1
$$

then:

$$
L(1, \hat{y}) = -\log(\hat{y})
$$

So:
- if \(\hat{y}\) is close to `1`, the loss is small
- if \(\hat{y}\) is close to `0`, the loss becomes very large

This means the model is strongly penalized when it assigns very low probability to the true positive class.

#### Case 2 — true class is 0

If:

$$
y = 0
$$

then:

$$
L(0, \hat{y}) = -\log(1-\hat{y})
$$

So:
- if \(\hat{y}\) is close to `0`, the loss is small
- if \(\hat{y}\) is close to `1`, the loss becomes very large

This means the model is strongly penalized when it assigns very high probability to the wrong class.

### Why this loss is useful

Cross-entropy is useful because it measures not only whether the predicted class is correct, but also how confident the model is.

This is important in classification.

For example, if the true class is `1`:
- predicting `0.55` is better than predicting `0.2`
- predicting `0.95` is better than predicting `0.55`

So cross-entropy rewards confident correct predictions and penalizes confident wrong predictions.

### Intuition

Cross-entropy behaves like a confidence-sensitive penalty.

#### Correct and confident prediction

If the model assigns a high probability to the true class, the loss is small.

#### Wrong and confident prediction

If the model assigns a high probability to the wrong class, the loss becomes very large.

This makes cross-entropy especially suitable for probability-based classification models.

### Example values

#### True label 1, predicted probability 0.9

$$
L(1, 0.9) = -\log(0.9) \approx 0.105
$$

Small loss, because the model is correct and confident.

#### True label 1, predicted probability 0.1

$$
L(1, 0.1) = -\log(0.1) \approx 2.3026
$$

Large loss, because the model is strongly wrong.

#### True label 0, predicted probability 0.1

$$
L(0, 0.1) = -\log(0.9) \approx 0.105
$$

Again, small loss.

#### True label 0, predicted probability 0.9

$$
L(0, 0.9) = -\log(0.1) \approx 2.3026
$$

Again, large loss.

These examples show the symmetry of the loss across the two classes.

### Dataset-level loss

For a dataset with \(n\) samples, the average binary cross-entropy is:

$$
\mathrm{BCE} = -\frac{1}{n}\sum_{i=0}^{n-1}\left[y_i\log(\hat{y}_i) + (1-y_i)\log(1-\hat{y}_i)\right]
$$

This is the natural full-dataset loss for Logistic Regression.

### Important practical note

Because cross-entropy contains:
- \(\log(\hat{y})\)
- \(\log(1-\hat{y})\)

it is numerically problematic if:
- \(\hat{y} = 0\)
- \(\hat{y} = 1\)

because the logarithm would become undefined.

In practice, implementations usually protect against this with small numeric safeguards.

For the first theory view of the project, the important idea is simply:
- cross-entropy expects probability-like inputs strictly inside `(0, 1)`

### Why this matters for the project

In the current project, cross-entropy is important because it provides the first classification-specific training objective.

It creates the bridge between:
- sigmoid outputs
- binary targets
- optimization
- later gradient derivation

Without cross-entropy, Logistic Regression would still have outputs and predictions, but it would not yet have its natural learning objective.

### Current implementation status

At the current stage of the project, the first binary cross-entropy utilities are implemented.

The current project supports:
- one-sample BCE through `ml::common::binary_cross_entropy(double y_true, double y_pred)`
- dataset-level average BCE through `ml::common::binary_cross_entropy(const std::vector<double>& y_true, const std::vector<double>& y_pred)`
- validation for invalid binary labels
- validation for probability-like inputs outside `(0, 1)`
- validation for empty vectors and size mismatch in the dataset-level version

So cross-entropy is no longer only a theoretical classification loss in the project. It is now also an implemented loss utility.

### First scope in this project

For the first version, cross-entropy should stay simple:
- binary labels only
- scalar probability for one-sample understanding
- average BCE for dataset-level training
- clean connection to sigmoid output

That is enough for a strong first implementation.

### Current empirical observations

The current implementation already shows the expected behavior on simple binary examples.

#### Correct and confident positive prediction

For:

$$
y = 1, \quad \hat{y} = 0.9
$$

The loss is approximately:

$$
0.105361
$$

This is small, which matches the idea that correct and confident predictions should be rewarded.

#### Wrong and confident positive prediction

For:

$$
y = 1, \quad \hat{y} = 0.1
$$

The loss is approximately:

$$
2.30259
$$

This is large, which matches the idea that confident wrong predictions should be penalized strongly.

#### Correct and confident negative prediction

For:

$$
y = 0, \quad \hat{y} = 0.1
$$

The loss is again approximately:

$$
0.105361
$$

#### Wrong and confident negative prediction

For:

$$
y = 0, \quad \hat{y} = 0.9
$$

The loss is again approximately:

$$
2.30259
$$

These results confirm the class-symmetric behavior of BCE.

#### Small dataset example

For:

$$
y_{\text{true}} = [1, 0], \quad y_{\text{pred}} = [0.9, 0.1]
$$

The average BCE is approximately:

$$
0.105361
$$

This confirms that the dataset-level implementation behaves consistently with the one-sample version.

#### Invalid-label case

For an invalid binary label such as:

$$
y = 2
$$

with a valid probability-like prediction, the implementation throws an exception.

#### Invalid-probability case

For an invalid probability-like prediction such as:

$$
\hat{y} = 0
$$

or

$$
\hat{y} = 1
$$

the implementation throws an exception.

These results match the intended first-scope behavior of the project.

## Cross-entropy as a classification loss utility

At this stage of the project, cross-entropy is now represented at two levels:
- as a theoretical classification loss
- as a concrete reusable utility in the common module

This is important because Logistic Regression now has:
- a bounded probability-like output through sigmoid
- a binary prediction interpretation through thresholding
- a classification-specific loss through BCE

So the project now has the essential output-and-loss layer needed before implementing the full Logistic Regression training loop.

---

## Gradient and training loop for Logistic Regression

### Purpose

Once Logistic Regression has:
- a linear score
- a sigmoid output
- a binary interpretation
- a cross-entropy loss

it becomes possible to define how the model learns from data.

This concept explains how Logistic Regression updates its parameters:
- weight
- bias

so that the binary cross-entropy loss becomes smaller over time.

So this is the first full training mechanism of the Logistic Regression phase.

### The model pipeline during training

For one sample, Logistic Regression follows this sequence:

#### 1. Linear score

$$
z = wx + b
$$

#### 2. Sigmoid output

$$
\hat{y} = \sigma(z)
$$

#### 3. Cross-entropy loss

$$
L(y, \hat{y}) = -\left[y\log(\hat{y}) + (1-y)\log(1-\hat{y})\right]
$$

So the model does not update the parameters directly from \(x\) and \(y\).

It updates them through the loss.

That is why gradients are needed.

### Why gradients are needed

The goal of training is to reduce the loss by changing the parameters.

So the model must answer:
- how does the loss change if `w` changes?
- how does the loss change if `b` changes?

These questions are answered by the gradients:

$$
\frac{\partial L}{\partial w}, \quad \frac{\partial L}{\partial b}
$$

Once these are known, Gradient Descent can be used to update the parameters.

### Key simplification for Logistic Regression

A very important result in Logistic Regression is that when sigmoid output is combined with binary cross-entropy loss, the gradient simplifies cleanly.

For one sample, the prediction error term becomes:

$$
\hat{y} - y
$$

This leads to the gradients:

$$
\frac{\partial L}{\partial w} = (\hat{y} - y)x
$$

$$
\frac{\partial L}{\partial b} = \hat{y} - y
$$

This is one of the most important formulas in the whole Logistic Regression phase.

It shows that the gradient depends on:
- the prediction error \((\hat{y} - y)\)
- the input value \(x\)

### Intuition for the gradient form

If the model predicts a probability that is too high relative to the true class, then:

$$
\hat{y} - y > 0
$$

So the update should push the parameters in the opposite direction.

If the model predicts a probability that is too low relative to the true class, then:

$$
\hat{y} - y < 0
$$

So the update direction changes accordingly.

This makes the error term:

$$
\hat{y} - y
$$

play the same central role in Logistic Regression that residual error plays in Linear Regression.

### Batch gradients for a dataset

For a dataset with \(n\) samples, the first version of the project should use full-batch Gradient Descent.

That means the gradients are averaged over the whole dataset.

So the batch gradients are:

$$
\frac{\partial \mathrm{BCE}}{\partial w} = \frac{1}{n}\sum_{i=0}^{n-1}(\hat{y}_i - y_i)x_i
$$

$$
\frac{\partial \mathrm{BCE}}{\partial b} = \frac{1}{n}\sum_{i=0}^{n-1}(\hat{y}_i - y_i)
$$

These are the natural dataset-level gradients for the first Logistic Regression training loop.

### Gradient Descent update rule

Once the gradients are known, the parameters are updated with learning rate \(\alpha\):

$$
w \leftarrow w - \alpha \frac{\partial \mathrm{BCE}}{\partial w}
$$

$$
b \leftarrow b - \alpha \frac{\partial \mathrm{BCE}}{\partial b}
$$

This is fully parallel to the update logic already used in Linear Regression.

The main difference is that now:
- predictions pass through sigmoid
- the loss is BCE instead of MSE
- the gradients are based on classification error in probability form

### Full training loop

For the first version of Logistic Regression, the training loop should remain simple and explicit.

For each epoch:
1. compute predictions for all samples
2. compute batch gradients
3. update `w` and `b`
4. optionally record the BCE loss and parameter values

So training is still an iterative optimization process, just like in Linear Regression.

### Why batch training is the right first choice

At this stage, the cleanest first version is batch Gradient Descent because:
- it is deterministic
- it is easier to debug
- it is easier to compare with the existing Linear Regression training loop
- it keeps the Logistic Regression phase focused on classification rather than optimizer variants

So the first training loop should be batch only.

### Relationship with the decision boundary

Because Logistic Regression uses:

$$
\hat{y} = \sigma(wx+b)
$$

and binary prediction is based on thresholding at `0.5`, training is really learning how to move the linear score:

$$
wx+b
$$

so that samples from the two classes fall on the correct side of the decision boundary.

This means the training loop is not only learning probabilities.

It is also learning the linear separation rule behind those probabilities.

### Why this matters for the project

In the current project, the Logistic Regression training loop is important because it creates the bridge between:
- classification theory
- implemented sigmoid
- implemented BCE
- optimization
- later prediction and evaluation workflow

Without this step, Logistic Regression would have all its pieces separately, but not yet a learning mechanism.

With this step, the model becomes trainable.

### Current implementation status

At the current stage of the project, the first Logistic Regression training-loop components are implemented.

The current project supports:
- scalar probability prediction through `predict_proba(double x)`
- vector probability prediction through `predict_proba(const std::vector<double>& x_values)`
- batch gradient computation through `gradient(...)`
- one batch Gradient Descent update through `gradient_descent_step(...)`
- repeated batch training through `fit_batch(...)`

So the Logistic Regression phase is no longer limited to output interpretation and BCE loss. It now includes a first trainable model implementation.

### First scope in this project

For the first version, the training loop should stay simple:
- one-dimensional input
- binary labels only
- full-batch Gradient Descent
- explicit gradients
- explicit parameter updates
- no regularization yet
- no SGD yet

That is enough for a strong first implementation.

### Current empirical observations

The current implementation already shows the expected behavior on a very small binary dataset.

#### Initial probability behavior

With:
- `weight = 0`
- `bias = 0`

all predicted probabilities are:

$$
0.5
$$

This matches the expected property that when the linear score is zero, sigmoid returns `0.5`.

#### Gradient sanity check

For the tiny dataset:

$$
x = [0, 1], \quad y = [0, 1]
$$

with initial parameters equal to zero, the observed batch gradient is:

$$
\frac{\partial \mathrm{BCE}}{\partial w} = -0.25
$$

$$
\frac{\partial \mathrm{BCE}}{\partial b} = 0
$$

This matches the expected first gradient calculation for the current project.

#### One-step update behavior

After one Gradient Descent step with learning rate `0.01`, the weight moves from:

$$
0 \rightarrow 0.0025
$$

while the bias remains:

$$
0
$$

This confirms that the update rule is using the gradient in the correct direction.

#### BCE improvement during training

For the same tiny dataset, the BCE before training is approximately:

$$
0.693147
$$

After batch training, the BCE decreases to approximately:

$$
0.680942
$$

So the model is moving in the correct learning direction, even though the improvement is small.

#### Probability movement after training

After training, the predicted probabilities move from:

$$
[0.5, 0.5]
$$

to approximately:

$$
[0.499855, 0.512206]
$$

This is consistent with the expected behavior:
- the sample from class `0` moves slightly below `0.5`
- the sample from class `1` moves slightly above `0.5`

So the classifier begins to separate the two classes in the correct direction.

These observations match the intended first-scope behavior of the project.

### Important practical note

The most important conceptual result to remember is:

$$
\hat{y} - y
$$

This term is the core error signal of Logistic Regression training.

Once that term is understood, the training loop becomes much easier to follow.

## Logistic Regression training as a model utility

At this stage of the project, the Logistic Regression training loop is now represented at two levels:
- as a theoretical optimization mechanism for binary classification
- as a concrete trainable model utility in the Logistic Regression module

This is important because the project now has:
- sigmoid outputs
- binary interpretation
- BCE loss
- batch gradients
- parameter updates
- repeated training across epochs

So the project now has the essential core of a first trainable binary classifier.

---

## Predictions

### Purpose

Once Logistic Regression is trainable, the final question is how the trained model should be used to make predictions.

At this stage, the project already has:
- sigmoid outputs
- binary interpretation
- cross-entropy loss
- a training loop

So this concept defines the final user-facing prediction behavior of the model.

It explains how Logistic Regression should produce:
- probability-like predictions
- binary class predictions

from learned parameters.

### Two prediction levels in the model

For the first version of Logistic Regression, the model should support two levels of prediction.

#### 1. Probability prediction

The model first computes the linear score:

$$
z = wx + b
$$

Then it applies the sigmoid:

$$
\hat{y} = \sigma(wx+b)
$$

This produces a probability-like output in:

$$
(0,1)
$$

So the first prediction level is not yet a class label.

It is a probability-like score expressing how strongly the model favors class `1`.

#### 2. Class prediction

To obtain a final binary decision, the probability-like output must be thresholded.

For the first version of the project, the prediction rule should remain:

$$
\hat{c} =
\begin{cases}
1 & \text{if } \hat{y} \ge 0.5 \\
0 & \text{if } \hat{y} < 0.5
\end{cases}
$$

So Logistic Regression should expose both:
- the probability-like prediction
- the thresholded binary prediction

### Why both prediction types matter

These two outputs serve different purposes.

#### Probability prediction is useful because:
- it preserves uncertainty information
- it is the natural input to BCE
- it shows how confident the model is

#### Class prediction is useful because:
- it gives the final classifier decision
- it is needed for later metrics such as accuracy and confusion matrix
- it makes the model usable as a classifier in practice

So a complete Logistic Regression model should provide both prediction types.

### Relationship with the decision boundary

Because class prediction is based on thresholding at `0.5`, and sigmoid is monotonic, the prediction rule is equivalent to:

$$
wx+b \ge 0 \Rightarrow \text{predict class } 1
$$

$$
wx+b < 0 \Rightarrow \text{predict class } 0
$$

This means the trained model is ultimately learning a linear decision boundary.

So even though the output probability is nonlinear, the class separation rule is still linear in the input space.

### Predictions before and after training

Before training, if the parameters are initialized at:

$$
w = 0, \quad b = 0
$$

then:

$$
\hat{y} = \sigma(0) = 0.5
$$

for every input.

So before training, the model is maximally uncertain.

After training, predictions should move away from `0.5` in a class-consistent direction:
- samples from class `0` should move toward `0`
- samples from class `1` should move toward `1`

This makes prediction behavior one of the clearest ways to observe whether learning is working.

### Vector predictions

For the first implementation, predictions should work at two input scales:

#### Scalar prediction

Given one input value:

$$
x
$$

the model should be able to return:
- one probability-like prediction
- one class prediction

#### Vector prediction

Given a vector of input values, the model should be able to return:
- a vector of probability-like predictions
- a vector of class predictions

This keeps the API aligned with the style already used in Linear Regression.

### Why this matters for the project

In the current project, predictions are important because they complete the first full Logistic Regression workflow:
- score computation
- sigmoid transformation
- binary interpretation
- BCE loss
- training
- final inference

Without this step, the model is trainable but not yet fully exposed as a classifier utility.

With this step, Logistic Regression becomes a complete first binary classifier in the project.

### First scope in this project

For the first version, predictions should stay simple:
- one-dimensional input
- probability prediction through sigmoid
- class prediction through fixed threshold `0.5`
- scalar and vector overloads

That is enough to complete the first Logistic Regression implementation.

### Important practical note

Later in Machine Learning, prediction thresholds can be adjusted depending on the application.

But for the first version of this project, using `0.5` as the fixed threshold is the correct and cleanest choice.