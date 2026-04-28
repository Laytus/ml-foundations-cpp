# k-NN

## Purpose

The goal of this sub-phase is to build the first non-parametric model in the project.

At this stage of the project, the previous models have been parametric:
- Linear Regression learns parameters such as weight and bias
- Logistic Regression also learns parameters such as weight and bias

k-NN is different.

Instead of learning a compact parameterized rule, it makes predictions by comparing a query point to stored training examples.

This sub-phase is important because it introduces:
- non-parametric learning
- instance-based prediction
- distance as the core comparison tool
- local decision behavior instead of a global learned equation

So k-NN is an important contrast with the models already implemented in the project.

---

## Why k-NN matters

So far, the project has focused on models that learn explicit parameters and then use those parameters to make predictions.

k-NN takes a different approach.

Instead of summarizing the training data into a small set of learned parameters, it keeps the training examples and compares new inputs directly against them.

This is important because it introduces a different way to think about Machine Learning:
- prediction by similarity
- local structure in the data
- decision making based on nearby examples

So k-NN helps broaden the project's ML intuition beyond gradient-based models.

---

## Distance

### Purpose

Distance is the first and most important mathematical concept in k-NN.

Its role is to measure how similar or how close two points are.

Because k-NN predicts by looking at nearby training examples, the model needs a clear rule for deciding what “nearby” means.

That rule is the distance function.

### Core idea

If two points are very close, then they should have a small distance.

If two points are far apart, then they should have a large distance.

So distance transforms geometric closeness into a numeric quantity that the model can compare.

This is the basic mechanism that allows k-NN to rank neighbors.

### One-dimensional distance

In one dimension, the simplest Euclidean distance between two values \(a\) and \(b\) is:

$$
d(a,b) = |a-b|
$$

This means the distance is just the absolute difference between the two values.

#### Example

If:

$$
a = 2, \quad b = 5
$$

then:

$$
d(2,5) = |2-5| = 3
$$

So the two points are `3` units apart.

### Multi-dimensional Euclidean distance

For vectors, the most standard first choice is Euclidean distance.

If:

$$
\mathbf{x} = (x_0, x_1, \dots, x_{n-1})
$$

and:

$$
\mathbf{z} = (z_0, z_1, \dots, z_{n-1})
$$

then the Euclidean distance is:

$$
d(\mathbf{x}, \mathbf{z}) = \sqrt{\sum_{i=0}^{n-1}(x_i - z_i)^2}
$$

This formula compares the vectors component by component, squares the differences, sums them, and then takes the square root.

So Euclidean distance measures straight-line distance in feature space.

### Example in two dimensions

Suppose:

$$
\mathbf{x} = (1,2)
$$

and:

$$
\mathbf{z} = (4,6)
$$

Then:

$$
d(\mathbf{x}, \mathbf{z}) = \sqrt{(1-4)^2 + (2-6)^2}
$$

$$
= \sqrt{(-3)^2 + (-4)^2}
$$

$$
= \sqrt{9 + 16}
$$

$$
= \sqrt{25} = 5
$$

So the two points are `5` units apart.

### Why Euclidean distance is a good first choice

For the first version of the project, Euclidean distance is the best starting point because:
- it is standard
- it is intuitive
- it is easy to compute
- it fits naturally with geometric intuition

Later in Machine Learning, other distance functions can also be useful.

But for the current project, Euclidean distance is the correct first distance metric.

### Why distance matters for k-NN

In k-NN, predictions depend directly on which training examples are closest to the query point.

So if the distance function changes, the selected neighbors can also change.

That means the distance function is not a small implementation detail.

It is one of the core design choices of the model.

### Relationship with feature scaling

Distance-based models are very sensitive to feature scale.

If one feature has a much larger numerical range than another, it can dominate the Euclidean distance.

So the feature-scaling work already done in the project becomes directly relevant again here.

This is an important connection between the earlier preprocessing phase and the new k-NN phase.

### Why this matters for the project

In the current project, distance is important because it creates the bridge between:
- stored training examples
- geometric similarity
- neighbor ranking
- later k-NN predictions

Without distance, k-NN cannot decide which samples are nearest.

With distance, the model can begin to compare and rank examples.

### Current implementation status

At the current stage of the project, the first distance utilities are implemented.

The current project supports:
- scalar Euclidean distance through `ml::common::euclidean_distance(double a, double b)`
- vector Euclidean distance through `ml::common::euclidean_distance(const std::vector<double>& x, const std::vector<double>& z)`
- validation for empty vectors in the vector version
- validation for size mismatch in the vector version

So distance is no longer only a theoretical concept in the k-NN phase. It is now also an implemented reusable utility.

### First scope in this project

For the first version, distance should stay simple:
- Euclidean distance only
- scalar and vector understanding
- reusable utility function
- clean validation for vector sizes

That is enough for a strong first implementation.

### Current empirical observations

The current implementation already shows the expected behavior on simple examples.

#### Scalar example

For:

$$
a = 2, \quad b = 5
$$

the observed Euclidean distance is:

$$
3
$$

This matches the expected one-dimensional distance:

$$
|2-5| = 3
$$

#### Scalar symmetry check

For:

$$
a = 5, \quad b = 2
$$

the observed Euclidean distance is also:

$$
3
$$

This confirms the expected symmetry of Euclidean distance.

#### Vector example

For:

$$
\mathbf{x} = [1,2]
$$

and:

$$
\mathbf{z} = [4,6]
$$

the observed Euclidean distance is:

$$
5
$$

This matches the expected result from the worked example in the theory section.

#### Identical-vectors case

For:

$$
\mathbf{x} = [1,2,3]
$$

and:

$$
\mathbf{z} = [1,2,3]
$$

the observed Euclidean distance is:

$$
0
$$

This matches the expected property that identical vectors have zero distance.

#### Size-mismatch case

When the two vectors have different sizes, the implementation throws an exception.

#### Empty-vector case

When the input vectors are empty, the implementation throws an exception.

These observations match the intended first-scope behavior of the project.

### Important practical note

For vector distance, both vectors must have the same dimension.

So the implementation will need to reject size mismatches.

That is not only a coding detail. It is part of the mathematical meaning of the distance itself.

---

## Distance as a reusable utility

At this stage of the project, distance is now represented at two levels:
- as a theoretical similarity measure for k-NN
- as a concrete reusable utility in the common module

This is important because the project now has:
- a first geometric comparison tool
- scalar and vector Euclidean distance
- validation for invalid vector inputs

So the project now has the essential first building block needed before moving to neighbor ranking, k selection, and k-NN predictions.

---

## k selection

### Purpose

Once a distance function exists, the next question in k-NN is how many neighbors should participate in the prediction.

That quantity is:

$$
k
$$

So k selection defines how large the local neighborhood should be around the query point.

This is one of the most important design choices in the whole k-NN model.

### Core idea

The k-NN rule is simple in principle:
- compute the distance from the query point to all training examples
- sort the training examples by distance
- keep the `k` nearest neighbors
- make the prediction from those neighbors

So `k` controls how much local information the model uses.

If `k` is small, the prediction depends on a very small neighborhood.

If `k` is large, the prediction depends on a broader neighborhood.

### Why k matters

The value of `k` strongly affects model behavior.

It changes:
- how local the prediction is
- how sensitive the model is to noise
- how smooth or irregular the decision rule becomes

So `k` is not a minor implementation detail.

It is one of the main factors controlling the behavior of k-NN.

### Small k

If `k` is very small, the model becomes highly local.

#### Example: \(k = 1\)

With:

$$
k = 1
$$

k-NN predicts using only the single nearest neighbor.

This makes the model very responsive to local structure.

But it also makes the model sensitive to:
- noisy training points
- outliers
- accidental local irregularities

So very small `k` can produce unstable predictions.

### Large k

If `k` is larger, the prediction uses more neighbors.

This makes the model less sensitive to individual noisy points.

But if `k` becomes too large, the model may lose important local structure and become overly coarse.

So large `k` can make predictions too global.

### Local vs global intuition

So there is a basic trade-off:

#### Small k
- more local
- more flexible
- more sensitive to noise

#### Large k
- more stable
- smoother behavior
- less sensitive to local detail

This is one of the key intuitions of k-NN.

### Why odd k is often preferred in binary classification

For binary classification, an odd value of `k` is often preferred.

The reason is simple:
- with odd `k`, majority voting is less likely to produce a tie

For example:
- `k = 3` gives a clearer majority vote than `k = 2`
- `k = 5` gives a clearer majority vote than `k = 4`

So for the first version of binary k-NN in this project, odd `k` values are the cleanest default choice.

### Constraint on k

For a training dataset with \(n\) samples, `k` must satisfy:

$$
1 \le k \le n
$$

because the model cannot choose more neighbors than there are training samples.

So any implementation must validate that `k` stays inside that range.

### k selection in practice

There is no universally best value of `k`.

The best choice depends on:
- the dataset
- the noise level
- the local structure of the classes
- the feature representation

So in practice, `k` is usually treated as a model hyperparameter.

For the first version of the project, the goal is not to optimize `k` exhaustively.

The goal is to understand clearly how changing `k` changes prediction behavior.

### Why this matters for the project

In the current project, k selection is important because it creates the bridge between:
- distance computation
- neighbor ranking
- final prediction behavior

Without `k`, the model can measure closeness, but it still cannot decide how many neighbors to use.

With `k`, the project gains the core neighborhood-size control of k-NN.

### Current implementation status

At the current stage of the project, the first `k`-selection utility is implemented.

The current project supports:
- `k` validation through `ml::is_valid_k(int k, std::size_t n_samples)`
- acceptance of values satisfying:
  - `k >= 1`
  - `k <= n_samples`
- rejection of invalid values such as:
  - `k <= 0`
  - `k > n_samples`
  - any `k` when `n_samples = 0`

So k selection is no longer only a theoretical control parameter in the k-NN phase. It now also has a first implemented validation utility.

### First scope in this project

For the first version, k selection should stay simple:
- integer `k` only
- binary classification focus
- require `1 <= k <= n`
- prefer odd `k` in examples
- no automated hyperparameter tuning yet

That is enough for a strong first implementation.

### Current empirical observations

The current implementation already shows the expected behavior on simple `k`-validation examples.

#### Smallest valid case

For:

$$
k = 1, \quad n = 5
$$

the observed validity result is:

$$
\text{true}
$$

This matches the expected rule that the smallest valid neighborhood size is `1`.

#### Largest valid case

For:

$$
k = 5, \quad n = 5
$$

the observed validity result is:

$$
\text{true}
$$

This matches the expected rule that `k` may equal the number of available training samples.

#### Middle valid case

For:

$$
k = 3, \quad n = 5
$$

the observed validity result is:

$$
\text{true}
$$

This confirms that valid interior values are accepted as expected.

#### Zero-k case

For:

$$
k = 0, \quad n = 5
$$

the observed validity result is:

$$
\text{false}
$$

This matches the intended rule that `k` must be at least `1`.

#### Negative-k case

For:

$$
k = -1, \quad n = 5
$$

the observed validity result is:

$$
\text{false}
$$

This matches the intended rule that negative neighborhood sizes are invalid.

#### Too-large-k case

For:

$$
k = 6, \quad n = 5
$$

the observed validity result is:

$$
\text{false}
$$

This matches the rule that `k` cannot exceed the number of available training samples.

#### Empty-training-set case

For:

$$
k = 1, \quad n = 0
$$

the observed validity result is:

$$
\text{false}
$$

This matches the intended behavior that no positive value of `k` is valid when there are no training samples.

These observations match the intended first-scope behavior of the project.

### Important practical note

k selection is tightly connected to distance.

If the distance ranking changes, the selected neighbors also change.

So `k` should not be studied separately from the distance function.

The two ideas work together to determine the final k-NN behavior.

---

## k selection as a model control parameter

At this stage of the project, k selection is now represented at two levels:
- as a theoretical neighborhood-size control concept in k-NN
- as a concrete reusable validation utility in the k-NN module

This is important because the project now has:
- distance as a comparison tool
- a validation rule for acceptable neighborhood size
- the first control layer needed before neighbor selection and prediction

So k selection now acts as the bridge between raw distance computation and the later prediction mechanism of k-NN.

---

## Predictions

### Purpose

Once distance and valid `k` selection exist, the final step is to define how k-NN actually makes predictions.

At this stage, the project already has:
- distance as the comparison rule
- `k` as the neighborhood-size control parameter

So this concept explains how those pieces combine into a full prediction mechanism.

It defines how k-NN should produce a final class prediction from nearby training examples.

### Core prediction workflow

For the first version of k-NN, the prediction process should follow four steps.

#### 1. Compute distances

Given a query point, compute its distance to every training example.

#### 2. Rank neighbors

Sort the training examples by distance from smallest to largest.

#### 3. Select the `k` nearest neighbors

Keep only the first `k` examples after sorting.

These are the local neighborhood that will influence the prediction.

#### 4. Predict from the selected neighbors

For binary classification, the simplest first rule is majority voting.

So the predicted class is the class label that appears most often among the `k` nearest neighbors.

This is the core prediction rule of the first k-NN implementation in the project.

### Prediction by majority vote

For binary classification, each selected neighbor has a class label in:

$$
\{0, 1\}
$$

So once the `k` nearest neighbors are known, the model counts how many belong to each class.

The simplest prediction rule is:
- predict class `1` if more neighbors belong to class `1`
- predict class `0` if more neighbors belong to class `0`

So k-NN predicts by local voting, not by applying a learned equation.

This is one of the most important differences from Linear Regression and Logistic Regression.

### Example

Suppose the query point has the following `k = 3` nearest neighbor labels:

$$
[1, 0, 1]
$$

Then:
- class `1` appears `2` times
- class `0` appears `1` time

So the prediction is:

$$
1
$$

Now suppose the `k = 5` nearest neighbor labels are:

$$
[0, 1, 0, 0, 1]
$$

Then:
- class `0` appears `3` times
- class `1` appears `2` times

So the prediction is:

$$
0
$$

These examples show that the prediction comes entirely from the local neighborhood.

### Why predictions in k-NN are local

The prediction for a query point depends only on nearby training examples.

So unlike a global parametric model, k-NN does not compress all training data into a few learned parameters.

Instead, the model keeps the training examples and performs local reasoning at prediction time.

This means k-NN predictions are:
- data-dependent
- local
- strongly influenced by distance and `k`

### Relationship with distance

Predictions in k-NN depend directly on the distance function.

If the distance function changes, the ranking of neighbors may change.

If the ranking changes, the selected `k` nearest neighbors may also change.

So prediction behavior cannot be understood separately from the distance function.

Distance determines which examples are even allowed to vote.

### Relationship with k

Predictions in k-NN also depend directly on `k`.

If `k` is small, prediction is based on a very local neighborhood.

If `k` is larger, prediction reflects a broader neighborhood.

So changing `k` can change the final predicted class even when the query point and training data stay the same.

That is why k-NN predictions should be understood as the combined result of:
- distance
- neighbor ranking
- neighborhood size
- voting rule

### Why odd k helps again

For binary classification, odd `k` values are especially convenient because they reduce the chance of ties in majority voting.

For example:
- with `k = 3`, one class usually has a clear majority
- with `k = 4`, a `2-2` tie is possible

So for the first version of the project, odd `k` remains the cleanest choice for prediction examples.

### What the first implementation should and should not do

For the first version, k-NN prediction should stay simple:
- binary classification only
- Euclidean distance only
- majority vote only
- one query point at a time first
- no probability output yet
- no distance weighting yet
- no tie-breaking complexity beyond the simplest policy if needed later

That is enough for a strong first implementation.

### Why this matters for the project

In the current project, predictions are important because they complete the first full k-NN workflow:
- distance computation
- neighbor ranking
- `k` validation
- local voting
- final class prediction

Without prediction logic, the project has the components of k-NN, but not yet the model behavior itself.

With prediction logic, k-NN becomes a complete first non-parametric classifier in the project.

### Current implementation status

At the current stage of the project, the first k-NN prediction utilities are implemented.

The current project supports:
- a `KNNClassifier` model storing:
  - training inputs
  - training labels
  - `k`
- scalar prediction through `predict(double x)`
- vector prediction through `predict(const std::vector<double>& x_values)`
- constructor validation for:
  - empty training data
  - mismatched training sizes
  - invalid binary labels
  - invalid `k`
- empty-query validation in the vector prediction version
- a deterministic majority-vote policy with ties returning class `0`

So predictions are no longer only theoretical in the k-NN phase. They are now also implemented model behavior.

### Important practical note

Unlike earlier models in the project, k-NN does not mainly spend effort learning parameters during training.

Its main complexity appears at prediction time, because it must compare the query point against the stored training data.

This is a defining property of the model and an important contrast with the previous sub-phases.

### Current empirical observations

The current implementation already shows the expected behavior on simple one-dimensional binary examples.

#### Scalar prediction near a negative training point

For:
- `x_train = [1, 2, 5]`
- `y_train = [0, 0, 1]`
- `k = 1`
- query `x = 1.2`

the observed prediction is:

$$
0
$$

This matches the expected behavior because the nearest training point is `1`, whose class is `0`.

#### Scalar prediction near a positive training point

For:
- `x_train = [1, 2, 5]`
- `y_train = [0, 0, 1]`
- `k = 1`
- query `x = 4.8`

the observed prediction is:

$$
1
$$

This matches the expected behavior because the nearest training point is `5`, whose class is `1`.

#### Majority-vote case with `k = 3`

For:
- `x_train = [1, 2, 3]`
- `y_train = [0, 1, 1]`
- `k = 3`
- query `x = 2.1`

the observed prediction is:

$$
1
$$

This matches the expected majority-vote result from the labels `[0, 1, 1]`.

#### Vector prediction case

For:
- `x_train = [1, 2, 5]`
- `y_train = [0, 0, 1]`
- `k = 1`
- query vector `x_values = [1.2, 4.8]`

the observed predictions are:

$$
[0, 1]
$$

This matches the expected pointwise behavior of the scalar prediction rule.

#### Tie case with even `k`

For:
- `x_train = [1, 2, 4, 5]`
- `y_train = [0, 0, 1, 1]`
- `k = 4`
- query `x = 3`

the observed prediction is:

$$
0
$$

This matches the chosen first-version tie policy in which ties return class `0`.

#### Invalid-constructor cases

The current implementation throws exceptions when:
- `k = 0`
- `k > n_samples`
- `x_train` and `y_train` have different sizes
- `y_train` contains a label outside `{0, 1}`

These results match the intended constructor validation rules.

#### Empty-query-vector case

The current implementation throws an exception when the query vector passed to `predict(const std::vector<double>& x_values)` is empty.

These observations match the intended first-scope behavior of the project.

---

## Predictions as the final k-NN mechanism

At this stage of the project, predictions are now represented at two levels:
- as a theoretical local classification mechanism
- as a concrete reusable model implementation through `KNNClassifier`

This is important because the project now has:
- distance as a comparison rule
- validated `k` selection
- stored training examples and labels
- local majority-vote prediction

So predictions now complete the first k-NN implementation and close the sub-phase as a working non-parametric classifier.

## What comes next in this doc

The next concepts to add in this document are:
- extending k-NN experiments and model behavior analysis

So this document will progressively become the k-NN reference for the whole sub-phase, connecting distance, implemented reusable utilities, k selection, neighbor selection, prediction behavior, model control choices, and completed non-parametric model behavior.