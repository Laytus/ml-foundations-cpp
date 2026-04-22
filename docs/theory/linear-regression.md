# Linear Regression

## Purpose

Linear Regression is the first trainable ML model in this project.

Its role in this phase is to build a deep understanding of:
- prediction
- loss functions
- gradients
- parameter updates
- optimization intuition

This model is intentionally studied before more advanced models because it provides the clearest path to understanding Gradient Descent at a low level.

---

## Core idea

Linear Regression tries to model the relationship between an input and an output using a straight line.

For the current implementation, the model uses a **single feature** input.

The model predicts an output using:

$$
\hat{y} = wx + b
$$

Where:
- $x$ = input feature
- $\hat{y}$ = predicted output
- $w$ = weight
- $b$ = bias

The weight controls the slope of the line.
The bias controls the vertical shift of the line.

---

## Current implementation scope

The current implementation is intentionally limited to:
- one input feature
- one scalar prediction per sample
- parameters stored as two `double` values

Current internal parameters:
- `weight_`
- `bias_`

This keeps the model simple enough to derive everything manually.

---

## Prediction

### Mathematical form

For one input value $x$:

$$
\hat{y} = wx + b
$$

For a dataset, the model applies the same formula independently to each input.

### Implemented API

The project currently supports:
- scalar prediction
- vector prediction
- manual parameter setting
- reading current parameter values

This allows the model to be used first as a prediction-only object before adding training.

### Intuition

Prediction is the forward computation of the model.

Given current values of `weight_` and `bias_`, the model produces an output.

At this stage, the model does not yet decide whether the prediction is good or bad. It only computes it.

---

## Mean Squared Error (MSE)

### Purpose

MSE is the first loss function implemented for Linear Regression.

Its role is to measure how wrong the current model is on a dataset.

### Per-sample error

For one sample:

$$
e_i = \hat{y}_i - y_i
$$

Where:
- $\hat{y}_i$ = model prediction
- $y_i$ = true target value
- $e_i$ = prediction error

### Squared error

The squared error for one sample is:

$$
e_i^2 = (\hat{y}_i - y_i)^2
$$

Squaring is important because:
- negative and positive errors should not cancel each other
- larger errors should be penalized more strongly

### MSE formula

For a dataset with $n$ samples:

$$
\mathrm{MSE} = \frac{1}{n} \sum_{i=0}^{n-1}{(\hat{y}_i - y_i)^2}
$$

### Intuition

MSE is the average squared distance between predictions and targets.

- smaller MSE means better fit
- MSE equal to 0 means perfect predictions on the dataset

### Current implementation notes

The current implementation:
- requires `x` and `y` to have the same size
- requires the dataset to be non-empty
- throws `std::invalid_argument` on invalid input
- computes predictions using the current model parameters

### Sanity checks already tested

For dataset:
- `x = [1, 2, 3]`
- `y = [3, 5, 7]`

Case 1:
- `w = 2`
- `b = 1`

Then:
- predictions are exact
- MSE = `0`

Case 2:
- `w = 0`
- `b = 0`

Then:
- predictions = `[0, 0, 0]`
- squared errors = `9, 25, 49`
- MSE = `83 / 3 = 27.6667`

---

## Gradient

### Purpose

The gradient tells us how the loss changes when the model parameters change.

For Linear Regression in the current implementation, the model has two trainable parameters:
- weight
- bias

So the gradient has two components:

$$
\frac{\partial \mathrm{MSE}}{\partial w}
\qquad
\frac{\partial \mathrm{MSE}}{\partial b}
$$

### Starting point

The model predicts:

$$
\hat{y}_i = wx_i + b
$$

The per-sample error is:

$$
e_i = \hat{y}_i - y_i = (wx_i + b) - y_i
$$

And the loss is:

$$
\mathrm{MSE} = \frac{1}{n} \sum_{i=0}^{n-1}{e_i^2}
$$

### Gradient with respect to weight

We want:

$$
\frac{\partial \mathrm{MSE}}{\partial w} = \frac{\partial}{\partial w}
\left(\frac{1}{n} \sum_{i=0}^{n-1} \left(e_i^2\right)\right)
$$

Take constants out:

$$
\frac{\partial \mathrm{MSE}}{\partial w} = \frac{1}{n} \sum_{i=0}^{n-1}{\frac{\partial}{\partial w} (e_i^2)}
$$

Apply the chain rule:

$$
\frac{\partial}{\partial w} (e_i^2) = 2e_i \frac{\partial e_i}{\partial w}
$$

Now:

$$
e_i = wx_i + b - y_i
$$

so:

$$
\frac{\partial e_i}{\partial w} = x_i
$$

Therefore:

$$
\frac{\partial \mathrm{MSE}}{\partial w} = \frac{2}{n} \sum_{i=0}^{n-1}{e_i x_i}
$$

So in expanded form:

$$
\frac{\partial \mathrm{MSE}}{\partial w} = \frac{2}{n} \sum_{i=0}^{n-1}{(\hat{y}_i - y_i)x_i}
$$

### Gradient with respect to bias

Same process:

$$
\frac{\partial \mathrm{MSE}}{\partial b} = \frac{1}{n} \sum_{i=0}^{n-1} \frac{\partial}{\partial b}\left(e_i^2\right)
$$

Again:

$$
\frac{\partial}{\partial b} (e_i^2) = 2e_i \frac{\partial e_i}{\partial b}
$$

and since:

$$
e_i = wx_i + b - y_i
$$

we get:

$$
\frac{\partial e_i}{\partial b} = 1
$$

So:

$$
\frac{\partial \mathrm{MSE}}{\partial b} = \frac{2}{n} \sum_{i=0}^{n-1}{(\hat{y}_i - y_i)}
$$

### Intuition

The two gradient components do not behave the same way.

#### Weight gradient

The weight multiplies the input $x_i$, so changing the weight affects prediction proportionally to the input magnitude.

That is why $x_i$ appears in:

$$
\frac{\partial \mathrm{MSE}}{\partial w}
$$

#### Bias gradient

The bias is added equally to every prediction, so changing the bias shifts all predictions by the same amount.

That is why the bias gradient is based only on the sum of errors.

### Current implementation notes

The current implementation:
- requires `x` and `y` to have the same size
- requires a non-empty dataset
- returns a `std::pair<double, double>`
- uses:
    - first = gradient with respect to weight
    - second = gradient with respect to bias

### Sanity check already tested

For dataset:
- `x = [1, 2, 3]`
- `y = [3, 5, 7]`
- `w = 0`
- `b = 0`

Predictions:

$$
[0, 0, 0]
$$

Errors:

$$
[-3, -5, -7]
$$

Then:

$$
\sum e_i x_i = (-3)(1) + (-5)(2) + (-7)(3) = -34
$$

$$
\sum{e_i} = -15
$$

With $n = 3$:

$$
\frac{\partial \mathrm{MSE}}{\partial w} = \frac{2}{3}(-34) = -22.6667
$$

$$
\frac{\partial \mathrm{MSE}}{\partial b} = \frac{2}{3}(-15) = -10
$$

This matches the implemented result already tested in the project.

---


## Gradient Descent (GD) training

### Purpose

Gradient Descent is the first training procedure implemented for Linear Regression.

Its role is to update the model parameters step by step so that the loss decreases over time.

In this project, Gradient Descent is the first mechanism that turns Linear Regression from a prediction-only model into a trainable model.

### Core idea

The model has two trainable parameters:
- weight \(w\)
- bias \(b\)

The gradient tells us the direction in which the loss increases the most.

So, to reduce the loss, we update the parameters in the opposite direction of the gradient.

### Update rule

Let:
- \(\alpha\) = learning rate
- \(\frac{\partial \mathrm{MSE}}{\partial w}\) = gradient with respect to weight
- \(\frac{\partial \mathrm{MSE}}{\partial b}\) = gradient with respect to bias

Then one Gradient Descent update step is:

$$
w \leftarrow w - \alpha \frac{\partial \mathrm{MSE}}{\partial w}
$$

$$
b \leftarrow b - \alpha \frac{\partial \mathrm{MSE}}{\partial b}
$$

### Intuition

If a gradient component is positive, increasing that parameter would increase the loss, so the parameter should be moved downward.

If a gradient component is negative, increasing that parameter would decrease the loss, so subtracting the negative gradient moves the parameter upward.

This is why the update rule uses subtraction.


### One step vs full training

It is important to distinguish between:

#### One GD step

One step means:
1. compute the gradient at the current parameter values
2. update `weight_`
3. update `bias_`

This is the smallest training unit.

#### Full GD training loop

A full training loop means repeating the same update many times.

Conceptually:
1. compute gradient
2. update parameters
3. compute new gradient
4. update again
5. repeat for many iterations

So full training is just repeated application of one step.

### Full batch training loop (`fit_batch(...)`)

The next layer after one GD step is the full batch training loop.

Its role is to apply the same update repeatedly for a fixed number of epochs.

#### Purpose

While `gradient_descent_step(...)` performs only one parameter update, `fit_batch(...)` performs many updates in sequence.

This is what allows the model to progressively improve instead of changing parameters only once.

#### Core idea

At each epoch:
1. compute the gradient using the full dataset
2. update `weight_`
3. update `bias_`
4. move to the next epoch

So the full training loop is just repeated application of one GD step.

#### Conceptual form

If the model starts with initial parameters:
- \(w_0\)
- \(b_0\)

then after one epoch it becomes:
- \(w_1\)
- \(b_1\)

after two epochs:
- \(w_2\)
- \(b_2\)

and so on.

The parameters evolve through a sequence of updates:

$$
(w_0, b_0) \rightarrow (w_1, b_1) \rightarrow (w_2, b_2) \rightarrow \cdots
$$

The goal is that this sequence moves toward parameters with lower loss.

#### Why the full dataset is used each epoch

In batch Gradient Descent, each epoch uses all samples to compute one average gradient.

This means:
- the update direction is based on the whole dataset
- updates are more stable than SGD
- training is easier to reason about mathematically

That is why batch GD is the best first training loop to implement in this phase.

#### Role of the `epochs` parameter

The `epochs` parameter controls how many times the update is repeated.

- `epochs = 1` means one GD step
- larger values mean more training iterations

So `epochs` controls how long the optimization process runs.

#### Expected behavior

If the implementation is correct and the learning rate is reasonable, increasing the number of epochs should usually:
- reduce MSE over time
- move the parameters closer to better values
- improve predictions progressively

However, improvement is not guaranteed for arbitrary learning rates. If the learning rate is too large, the loss may oscillate or increase instead of decreasing.

#### Important note

`fit_batch(...)` should not introduce new optimization math.

Its purpose is to organize repetition.

The optimization logic already exists in:
- the gradient computation
- the one-step GD update rule

So `fit_batch(...)` is conceptually simple, but very important because it is the first complete training loop of the model.

### Learning rate

The learning rate controls how large each update is.

- if the learning rate is too small, training becomes very slow
- if the learning rate is too large, training can overshoot or diverge

So the learning rate controls the trade-off between speed and stability.

### Batch Gradient Descent in the current scope

The current implementation scope uses **batch Gradient Descent** first.

That means the gradient is computed using the full dataset before each update.

So for each update:
- all samples are used
- one average gradient is computed
- one parameter update is applied

This is different from SGD, which will come later.

### Current implementation status

At the current stage of the project:
- `gradient_descent_step(...)` is implemented
- `fit_batch(...)` is implemented

The current implemented GD layers are:

1. `gradient_descent_step(...)`
   - compute the current gradient
   - update the parameters once

2. `fit_batch(...)`
   - repeat the one-step update for many epochs

This separation remains useful because it makes the training process easier to understand and debug.

### Current implementation notes

The current implementation of `gradient_descent_step(...)`:
- requires `x` and `y` to have the same size
- requires a non-empty dataset
- requires `learning_rate > 0`
- computes the gradient using the current parameter values
- updates both parameters using the same gradient evaluation point

### Sanity check already tested

For dataset:
- `x = [1, 2, 3]`
- `y = [3, 5, 7]`
- initial `w = 0`
- initial `b = 0`
- learning rate `\alpha = 0.1`

Previously computed gradient:

$$
\frac{\partial \mathrm{MSE}}{\partial w} = -22.6667
$$

$$
\frac{\partial \mathrm{MSE}}{\partial b} = -10
$$

So after one GD step:

$$
w \leftarrow 0 - 0.1(-22.6667) = 2.26667
$$

$$
b \leftarrow 0 - 0.1(-10) = 1
$$

This matches the implemented result already tested in the project.

### Expected behavior

If the implementation is correct and the learning rate is reasonable, repeated GD steps should:
- reduce MSE over time
- move the parameters toward better values
- improve predictions

### Important note

Gradient Descent does not jump directly to the perfect solution in one update.

It is an iterative optimization process.

The parameters improve progressively through repeated small corrections.

This is one of the most important ideas in the whole phase.

---

## Convergence analysis for Linear Regression with batch GD
### Current implementation status

At the current stage of the project, the first convergence-analysis mechanism is implemented.

The model can now run batch GD while recording per-epoch training history.

This history includes, for each epoch:
- epoch number
- MSE
- weight
- bias

This makes it possible to inspect the dynamics of training directly from the program output.

### Purpose

Convergence analysis studies how the model behaves across repeated Gradient Descent updates.

The central question is:

> Does the training process move the model toward a better solution over time?

For Linear Regression with batch Gradient Descent, convergence analysis helps us observe whether:
- the loss decreases
- the parameters stabilize
- the updates behave smoothly
- the learning rate is appropriate

It is the first time we study not only one update, but the dynamics of many updates together.

### What convergence means here

In this context, convergence means that as training progresses:
- the parameter values move toward a stable region
- the loss tends to decrease
- successive updates become less dramatic
- the model stops making large corrections

For the current reference problem, the ideal target is the line:

$$
y = 2x + 1
$$

So convergence means the learned parameters should move toward something close to:
- \(w = 2\)
- \(b = 1\)

and the MSE should move toward:
- \(0\), or very close to it

### Why convergence matters

A training loop can exist and still behave badly.

For example:
- loss may decrease correctly
- loss may decrease too slowly
- loss may oscillate
- loss may explode
- parameters may diverge

So implementing Gradient Descent is only the first step.

Convergence analysis is what tells us whether the training process is actually healthy.

### Main quantities to observe

For the first convergence analysis in this project, the most important quantities are:

#### 1. MSE over epochs

This is the most important signal.

If training is working, MSE should generally decrease over time.

For a simple problem like the current one, with a reasonable learning rate, it should decrease clearly.

#### 2. Weight over epochs

Track how `weight_` changes through training.

This shows whether the slope is moving toward the expected value.

#### 3. Bias over epochs

Track how `bias_` changes through training.

This shows whether the intercept is moving toward the expected value.

### Expected qualitative behaviors

#### Case A: good convergence

If the learning rate is reasonable:
- MSE decreases steadily
- weight and bias approach stable values
- updates become smaller over time

This is the desired behavior.

#### Case B: very slow convergence

If the learning rate is too small:
- MSE decreases, but very slowly
- parameters improve only a little each epoch
- many epochs are needed

This is not incorrect, but inefficient.

#### Case C: oscillation or instability

If the learning rate is too large:
- MSE may jump up and down
- parameters may overshoot
- training may fail to settle

#### Case D: divergence

If the learning rate is much too large:
- MSE may increase instead of decrease
- parameters may grow in magnitude
- training becomes unstable

### Why batch GD is good for first convergence analysis

Batch Gradient Descent is ideal for this first study because:
- every epoch uses the full dataset
- the gradient is deterministic
- the behavior is easier to reason about
- noise from random sample selection is absent

That makes the convergence pattern much cleaner than SGD.

So batch GD is the right first environment to study training dynamics.

### Relationship between convergence and learning rate

The learning rate is one of the most important factors in convergence.

Recall the update rule:

$$
w \leftarrow w - \alpha \frac{\partial \mathrm{MSE}}{\partial w}
$$

$$
b \leftarrow b - \alpha \frac{\partial \mathrm{MSE}}{\partial b}
$$

where \(\alpha\) is the learning rate.

The learning rate controls the size of each correction:
- small \(\alpha\): careful but slow
- large \(\alpha\): fast but risky

So convergence analysis is deeply tied to learning rate analysis.

### First scope in this project

At the current stage, convergence analysis should stay simple and focused.

The first goal is not to do advanced optimization theory.

The first goal is to observe, in a practical and structured way:
- how loss changes with epochs
- how parameters move with epochs
- how learning rate affects this behavior

That is enough for a very strong first analysis.

### Current empirical observations

The current implementation already shows the main qualitative convergence cases expected from batch GD on the reference dataset.

#### Reasonable learning rate

With a reasonable learning rate such as `0.1`:
- MSE decreases very strongly
- parameters move quickly toward the target region
- training remains stable

#### Small learning rate

With a smaller learning rate such as `0.01`:
- MSE still decreases
- parameter updates are smaller
- convergence is slower

#### Large learning rate

With a large learning rate such as `1.0`:
- MSE explodes instead of decreasing
- parameters oscillate in sign and grow rapidly in magnitude
- training diverges

These observations match the theoretical expectations described earlier in this section.

### What this concept is not yet

For now, convergence analysis does not need to include:
- formal proofs
- Hessians
- full convex optimization theory
- line search methods
- momentum
- adaptive optimizers

Those belong much later.

At this stage, convergence analysis is primarily:
- empirical
- interpretable
- tied directly to the current implementation

### Core theoretical goal

The main theoretical idea to carry forward is:

> Gradient Descent is not just an update formula. It is a dynamic process over time.

Convergence analysis is the study of that process.

That is why it is such an important next step after implementing:
- MSE
- gradient
- one GD step
- full batch training loop

---

## What is implemented so far

At the current stage of the project, Linear Regression supports:
- parameter storage (`weight_`, `bias_`)
- scalar prediction
- vector prediction
- MSE computation
- gradient computation
- one Gradient Descent update step
- full batch training loop (`fit_batch(...)`)
- convergence analysis through training history recording
- getters for parameters
- manual parameter setting

Not implemented yet:
- stochastic gradient descent
- broader optimization experiments beyond the current batch GD analysis

---

## Why this model is important

Linear Regression is the first complete bridge between:
- mathematical modeling
- error measurement
- derivatives
- optimization
- code implementation

This makes it the ideal starting point for understanding how trainable ML models work internally.

It is not only a simple model. It is the foundation for understanding the rest of this phase, especially optimization, convergence, and training dynamics.