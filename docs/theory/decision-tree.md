# Decision Tree

## Purpose

The goal of this sub-phase is to build the first rule-based learning model in the project.

At this stage of the project, the previous models have followed other prediction styles:
- Linear Regression uses a continuous parametric equation
- Logistic Regression uses a parametric classification rule
- k-NN predicts by local similarity and neighbor voting

Decision Trees introduce a different idea.

Instead of predicting through a global equation or neighbor lookup, they make predictions by applying a sequence of simple decision rules.

This sub-phase is important because it introduces:
- rule-based learning
- recursive partitioning of the input space
- interpretable decisions based on thresholds
- hierarchical prediction structure

So Decision Trees add a new model family to the project with a very different style of reasoning.

---

## Why Decision Trees matter

Decision Trees are important because they model prediction as a sequence of explicit rules.

Instead of saying:
- “compute a weighted sum”
- or “look at the nearest examples”

they say:
- “check this condition”
- “go left or right”
- “repeat until reaching a final decision”

This makes them especially useful for understanding how Machine Learning can create human-readable decision processes.

So Decision Trees help broaden the project beyond:
- parametric models
- distance-based models

and into rule-based models.

---

## Splits

### Purpose

A split is the first and most important concept in a Decision Tree.

Its role is to divide a dataset into two smaller groups according to a rule.

This is the basic operation from which the whole tree is built.

So before studying impurity or recursive tree construction, the project first needs a clear understanding of what a split is and how it works.

### Core idea

A split takes a set of samples and separates them into two subsets.

For the first simplified version of the project, the split rule should be threshold-based.

That means the rule has the form:

$$
x < t
$$

where:
- \(x\) is the feature value
- \(t\) is a threshold

Then each sample is sent to one of two groups:
- left branch if the rule is true
- right branch if the rule is false

So a split turns one dataset into two smaller datasets.

### One-dimensional split rule

For the first version of the project, the cleanest starting point is one-dimensional data.

In that case, a split at threshold \(t\) should follow this rule:

- send the sample to the left subset if:

$$
x < t
$$

- send the sample to the right subset if:

$$
x \ge t
$$

This is the simplest decision-tree split rule and it is enough for the first implementation.

### Example

Suppose the input values are:

$$
[1, 2, 3, 5, 8]
$$

and the threshold is:

$$
t = 4
$$

Then the split becomes:

#### Left subset

Values satisfying:

$$
x < 4
$$

So the left subset is:

$$
[1, 2, 3]
$$

#### Right subset

Values satisfying:

$$
x \ge 4
$$

So the right subset is:

$$
[5, 8]
$$

This is the most basic structure behind a Decision Tree node.

### Why splits matter

A Decision Tree is built by applying splits repeatedly.

So a single split is not the full model.

But every node of the tree is based on exactly this operation:
- choose a rule
- divide the data
- continue on each side

That means the quality of the full tree depends directly on the quality of its splits.

So splits are not only the first step of the model. They are the core structural operation of the whole model.

### Geometric interpretation

A split creates a boundary in the input space.

In one dimension, that boundary is just a threshold point.

So a split divides the number line into two regions:
- values smaller than the threshold
- values greater than or equal to the threshold

Later, in higher dimensions, Decision Trees create axis-aligned decision boundaries by applying similar threshold rules feature by feature.

For the first version of the project, the one-dimensional interpretation is enough.

### Relationship with labels

The split rule itself depends on the input values, but its purpose is to improve how the labels are separated.

That means a split is useful if it tends to group similar labels together.

For example, if a threshold creates:
- mostly class `0` samples on one side
- mostly class `1` samples on the other side

then that split is likely helpful for classification.

This is the bridge to the next concept of the sub-phase:
- impurity

because impurity is what will later measure how good a split is.

### Why threshold-based splits are a good first choice

For the first version of the project, threshold-based splits are the best starting point because they are:
- simple
- interpretable
- easy to implement
- directly aligned with the standard Decision Tree idea

Later, more complex split strategies exist in Machine Learning.

But for the current project, threshold-based binary splits are the correct first scope.

### Why this matters for the project

In the current project, splits are important because they create the bridge between:
- raw input values
- rule-based data partitioning
- later impurity evaluation
- later tree construction

Without splits, there is no Decision Tree structure at all.

With splits, the project gains the first structural operation of rule-based learning.

### Current implementation status

At the current stage of the project, the first split utility is implemented.

The current project supports:
- threshold-based dataset splitting through `ml::split_dataset(const std::vector<double>& x_values, const std::vector<int>& y_values, double threshold)`
- a reusable `SplitResult` structure containing:
  - `left_x`
  - `left_y`
  - `right_x`
  - `right_y`
- validation for empty vectors
- validation for size mismatch
- validation for invalid binary labels in `y_values`

So splits are no longer only a theoretical concept in the Decision Tree phase. They are now also an implemented reusable utility.

### First scope in this project

For the first version, splits should stay simple:
- one-dimensional input
- one threshold at a time
- binary split only
- left branch for `x < t`
- right branch for `x >= t`

That is enough for a strong first implementation.

### Current empirical observations

The current implementation already shows the expected behavior on simple one-dimensional binary examples.

#### Mixed split case

For:

$$
x = [1, 2, 3, 5, 8]
$$

$$
y = [0, 0, 1, 1, 1]
$$

with threshold:

$$
t = 4
$$

the observed split is:
- `left_x = [1, 2, 3]`
- `left_y = [0, 0, 1]`
- `right_x = [5, 8]`
- `right_y = [1, 1]`

This matches the expected threshold rule and preserves label alignment correctly.

#### All-left case

For:

$$
x = [1, 2, 3]
$$

$$
y = [0, 1, 0]
$$

with threshold:

$$
t = 10
$$

the observed split sends all samples to the left subset and leaves the right subset empty.

This matches the expected behavior when every input satisfies:

$$
x < t
$$

#### All-right case

For:

$$
x = [5, 6, 7]
$$

$$
y = [1, 0, 1]
$$

with threshold:

$$
t = 2
$$

the observed split sends all samples to the right subset and leaves the left subset empty.

This matches the expected behavior when every input satisfies:

$$
x \ge t
$$

#### Threshold-equality case

For:

$$
x = [1, 4, 6]
$$

$$
y = [0, 1, 1]
$$

with threshold:

$$
t = 4
$$

the observed split is:
- `left_x = [1]`
- `left_y = [0]`
- `right_x = [4, 6]`
- `right_y = [1, 1]`

This confirms the intended rule:
- left for `x < t`
- right for `x >= t`

So a value exactly equal to the threshold goes to the right subset.

#### Size-mismatch case

When `x_values` and `y_values` have different sizes, the implementation throws an exception.

#### Empty-vector case

When the input vectors are empty, the implementation throws an exception.

#### Invalid-label case

When `y_values` contains a label outside the binary set `{0, 1}`, the implementation throws an exception.

These observations match the intended first-scope behavior of the project.

### Important practical note

A split should preserve the correspondence between:
- input values
- labels

So when a sample is sent left or right, its label must move with it.

This is not only an implementation detail. It is essential for later impurity computation and tree construction.

---

## Splits as a reusable utility

At this stage of the project, splits are now represented at two levels:
- as a theoretical rule-based partitioning mechanism
- as a concrete reusable utility in the Decision Tree module

This is important because the project now has:
- a threshold-based partition rule
- preserved alignment between inputs and labels
- a reusable left/right split representation

So the project now has the essential first structural operation needed before impurity evaluation and tree construction.

---

## Impurity

### Purpose

Once a split can divide the data into left and right subsets, the next question is whether that split is actually good.

A Decision Tree needs a way to measure how mixed or how pure the labels are inside a subset.

That is the role of impurity.

So impurity is the concept that allows the model to compare candidate splits and decide which one separates the labels better.

### Core idea

A subset is considered pure when most or all samples in it belong to the same class.

A subset is considered impure when the labels are strongly mixed.

So impurity is a numeric way to measure label mixing inside a subset.

This is one of the most important ideas in Decision Trees, because the model chooses splits that reduce impurity.

### Why impurity matters

A threshold-based split by itself is only a partition rule.

But a Decision Tree needs more than the ability to divide data.

It needs a way to answer:
- is this split useful?
- does this split separate the classes better than another one?

Impurity provides exactly that evaluation mechanism.

So impurity is what turns a split from a structural operation into a measurable decision rule.

### Intuition

Suppose a subset contains only class `0` samples.

Then that subset is very pure.

Suppose a subset contains roughly half class `0` and half class `1`.

Then that subset is much more impure.

So impurity is lowest when one class dominates and highest when the classes are strongly mixed.

### Gini impurity

For the first simplified Decision Tree in the project, the best impurity measure to start with is Gini impurity.

For a binary classification subset, if:

$$
p_0
$$

is the proportion of class `0`, and:

$$
p_1
$$

is the proportion of class `1`, then Gini impurity is:

$$
G = 1 - (p_0^2 + p_1^2)
$$

Since for binary classification:

$$
p_0 + p_1 = 1
$$

this is enough to describe the impurity of a node.

### Interpretation of Gini impurity

#### Perfectly pure subset

If all samples belong to one class, then one probability is `1` and the other is `0`.

For example:

$$
p_0 = 1, \quad p_1 = 0
$$

Then:

$$
G = 1 - (1^2 + 0^2) = 0
$$

So a perfectly pure subset has Gini impurity `0`.

#### Maximally mixed binary subset

If the two classes are evenly balanced, then:

$$
p_0 = 0.5, \quad p_1 = 0.5
$$

Then:

$$
G = 1 - (0.5^2 + 0.5^2)
$$

$$
= 1 - (0.25 + 0.25)
$$

$$
= 0.5
$$

So for binary classification, the maximum Gini impurity is `0.5`.

This means:
- Gini close to `0` → subset is pure
- Gini close to `0.5` → subset is highly mixed

### Example

Suppose a subset contains the labels:

$$
[0, 0, 1, 1]
$$

Then:

$$
p_0 = 0.5, \quad p_1 = 0.5
$$

So:

$$
G = 1 - (0.5^2 + 0.5^2) = 0.5
$$

Now suppose another subset contains:

$$
[0, 0, 0, 1]
$$

Then:

$$
p_0 = 0.75, \quad p_1 = 0.25
$$

So:

$$
G = 1 - (0.75^2 + 0.25^2)
$$

$$
= 1 - (0.5625 + 0.0625)
$$

$$
= 0.375
$$

So the second subset is purer than the first one.

### Impurity reduction and split quality

A split should be considered useful if it produces child subsets that are purer than the parent subset.

So the Decision Tree will later compare candidate thresholds by checking how much impurity is reduced after the split.

That means impurity itself is not yet the whole split-selection rule.

But it is the basic quantity from which split quality will be evaluated.

### Why Gini is a good first choice

For the first version of the project, Gini impurity is the best starting point because it is:
- standard
- intuitive
- easy to compute
- directly connected to binary class proportions

Later, Decision Trees can also use other impurity measures, such as entropy.

But for the current project, Gini impurity is the correct first scope.

### Relationship with splits

Splits divide the data.

Impurity measures the quality of the resulting groups.

So the two concepts work together:
- split gives the partition
- impurity tells whether that partition is useful

This is the central bridge between the first and second items of the Decision Tree sub-phase.

### Why this matters for the project

In the current project, impurity is important because it creates the bridge between:
- raw label distributions
- split evaluation
- later tree construction

Without impurity, the project can divide data, but it still cannot judge which split is better.

With impurity, the Decision Tree gains the first quantitative rule for choosing better partitions.

### Current implementation status

At the current stage of the project, the first impurity utility is implemented.

The current project supports:
- Gini impurity computation through `ml::gini_impurity(const std::vector<int>& y_values)`
- validation for empty label subsets
- validation for invalid binary labels in `y_values`

So impurity is no longer only a theoretical concept in the Decision Tree phase. It is now also an implemented reusable utility.

### First scope in this project

For the first version, impurity should stay simple:
- binary classification only
- Gini impurity only
- one subset at a time first
- dataset-level class proportions only

That is enough for a strong first implementation.

### Current empirical observations

The current implementation already shows the expected behavior on simple binary subsets.

#### Perfectly pure class `0` subset

For:

$$
y = [0, 0, 0, 0]
$$

the observed Gini impurity is:

$$
0
$$

This matches the expected impurity of a perfectly pure subset.

#### Perfectly pure class `1` subset

For:

$$
y = [1, 1, 1, 1]
$$

the observed Gini impurity is:

$$
0
$$

This also matches the expected impurity of a perfectly pure subset.

#### Maximally mixed subset

For:

$$
y = [0, 0, 1, 1]
$$

the observed Gini impurity is:

$$
0.5
$$

This matches the expected maximum Gini impurity for binary classification.

#### Partially mixed subset

For:

$$
y = [0, 0, 0, 1]
$$

the observed Gini impurity is:

$$
0.375
$$

This matches the expected value for a subset with class proportions \(0.75\) and \(0.25\).

#### Small mixed subset

For:

$$
y = [0, 1]
$$

the observed Gini impurity is:

$$
0.5
$$

This again matches the expected impurity of a perfectly balanced binary subset.

#### Empty-vector case

When `y_values` is empty, the implementation throws an exception.

#### Invalid-label case

When `y_values` contains a label outside the binary set `{0, 1}`, the implementation throws an exception.

These observations match the intended first-scope behavior of the project.

### Important practical note

Impurity depends only on the labels inside a subset, not on the raw input values directly.

The input values matter because they determine the split.

But once a subset is formed, impurity is computed from its class composition.

This is a very important conceptual separation:
- input values determine the partition
- labels determine the impurity

---

## Impurity as a split-evaluation mechanism

At this stage of the project, impurity is now represented at two levels:
- as a theoretical split-evaluation concept
- as a concrete reusable utility in the Decision Tree module

This is important because the project now has:
- threshold-based splitting
- a quantitative measure of label mixing
- the first numerical basis for comparing partitions

So impurity now turns splitting from a structural operation into a measurable learning rule.

---

## Tree structure

### Purpose

Once splits and impurity are available, the next step is to understand how a full Decision Tree is organized.

A tree is not just one split.

It is a hierarchical structure built by applying splits recursively until the data is separated well enough or until a stopping rule is reached.

So tree structure is the concept that turns individual split operations into a complete predictive model.

### Core idea

A Decision Tree is composed of nodes connected in a hierarchy.

For the first simplified version of the project, the structure should be understood through three main components:
- the root node
- internal decision nodes
- leaf nodes

Each node represents a subset of the data.

Each internal node applies a split rule.

Each leaf node stores a final prediction.

So the whole model can be seen as a recursive partitioning of the dataset into smaller and smaller regions.

### Root node

The root node is the starting point of the tree.

It contains the full training dataset before any split has been applied.

So the root node is where the first threshold decision is chosen.

After the root split, the dataset is divided into two child subsets:
- left child
- right child

This is the first level of the tree structure.

### Internal decision nodes

An internal node is a non-terminal node that still applies a split.

That means it receives a subset of the data and asks a threshold question such as:

$$
x < t
$$

Then it sends the subset further down the tree:
- left if the rule is true
- right if the rule is false

So each internal node represents another stage of rule-based partitioning.

A tree becomes deeper when this process is repeated multiple times.

### Leaf nodes

A leaf node is a terminal node.

It does not apply another split.

Instead, it stores the final prediction for the region of the input space that reaches that leaf.

For the first binary-classification version of the project, the simplest leaf behavior is:
- predict class `0`
- or predict class `1`

So a leaf node is the endpoint of a decision path.

### Decision path

A prediction in a Decision Tree is made by following a path from the root to a leaf.

At each internal node:
- evaluate the split rule
- choose left or right
- continue until reaching a leaf

So the prediction is the result of a sequence of local yes/no decisions.

This is the defining structural idea of the model.

### Example of a tiny tree

A very small one-dimensional binary Decision Tree could look like this:

- root split: `x < 4`
  - left child: predict `0`
  - right child: predict `1`

This means:
- if `x < 4`, output class `0`
- if `x >= 4`, output class `1`

Even this very small tree already has the full basic structure:
- one root node
- two leaves
- one decision rule

### Recursive construction intuition

A tree is usually built recursively.

That means:
1. choose a split for the current subset
2. create left and right child subsets
3. repeat the same logic on each child subset
4. stop when a terminal condition is reached

So the structure grows by repeating the same split logic on smaller and smaller subsets.

This is why the split and impurity concepts must come before tree structure.

### Why stopping rules matter

A tree cannot keep splitting forever.

So the model needs stopping rules.

For the first simplified project version, the stopping conditions should be understood conceptually, even if not all of them are implemented immediately.

Examples of stopping conditions are:
- the subset is already pure
- the subset is too small
- no useful split remains
- a maximum tree depth is reached

Once a stopping condition is met, the node becomes a leaf.

So leaf creation is tightly connected to stopping criteria.

### Relationship with impurity

Tree structure depends directly on impurity.

Impurity is what will later guide whether a split is worth making.

So the relationship is:
- split generates candidate partitions
- impurity evaluates them
- tree structure organizes the accepted splits into a hierarchy

This is the central bridge between the second and third items of the Decision Tree sub-phase.

### Why tree structure matters for the project

In the current project, tree structure is important because it creates the bridge between:
- local threshold decisions
- recursive partitioning
- final model prediction

Without tree structure, the project only has isolated split and impurity utilities.

With tree structure, the project gains the first full organization of those utilities into a complete rule-based classifier.

### Current implementation status

At the current stage of the project, the first explicit tree-node structure is implemented.

The current project supports:
- a reusable `DecisionTreeNode` structure containing:
  - `is_leaf`
  - `threshold`
  - `prediction`
  - `left`
  - `right`
- manual construction of:
  - leaf nodes for class `0`
  - leaf nodes for class `1`
  - a root split node connected to child leaves
- explicit pointer-based left/right child relationships
- manual prediction-path interpretation through the tiny-tree example

So tree structure is no longer only a theoretical concept in the Decision Tree phase. It now also has a first implemented structural representation.

### First scope in this project

For the first version, tree structure should stay simple:
- one-dimensional input
- binary classification only
- binary tree only
- threshold rule at each internal node
- class prediction at each leaf
- shallow structure first before deeper recursion

That is enough for a strong first implementation.

### Current empirical observations

The current implementation already shows the expected behavior on a tiny binary tree example.

#### Leaf node for class `0`

The project successfully constructs a leaf node with:
- `is_leaf = true`
- `prediction = 0`
- `left == nullptr`
- `right == nullptr`

This matches the intended structure of a terminal class-`0` leaf.

#### Leaf node for class `1`

The project successfully constructs a leaf node with:
- `is_leaf = true`
- `prediction = 1`
- `left == nullptr`
- `right == nullptr`

This matches the intended structure of a terminal class-`1` leaf.

#### Root split node

The project successfully constructs a root node with:
- `is_leaf = false`
- `threshold = 4`
- non-null left and right child pointers

This matches the intended structure of a non-terminal decision node.

#### Prediction-path case for `x = 2.5`

With the tiny tree:
- if `x < 4` → go left → predict `0`
- otherwise → go right → predict `1`

the observed prediction for:

$$
x = 2.5
$$

is:

$$
0
$$

This matches the expected left-branch behavior.

#### Prediction-path case for `x = 6`

For:

$$
x = 6
$$

the observed prediction is:

$$
1
$$

This matches the expected right-branch behavior.

#### Null-child check for leaves

The leaf nodes correctly satisfy:
- `left == nullptr`
- `right == nullptr`

This matches the intended first-scope leaf design.

These observations match the intended first-scope behavior of the project.

### Important practical note

A Decision Tree is easier to understand if the node structure is kept explicit.

So for the first implementation, it is better to represent clearly:
- whether a node is a leaf
- what threshold it uses if it is not a leaf
- what prediction it stores if it is a leaf
- which child nodes belong to it

This explicit structure will make debugging and later extensions much easier.

---

## Tree structure as the full Decision Tree mechanism

At this stage of the project, tree structure is now represented at two levels:
- as a theoretical hierarchical model structure
- as a concrete reusable node representation in the Decision Tree module

This is important because the project now has:
- threshold-based splitting
- impurity as a split-evaluation concept
- explicit leaf and internal-node structure
- a first path-based prediction interpretation

So tree structure now turns the existing Decision Tree utilities into the first concrete model architecture of the sub-phase.

## What comes next in this doc

The next concepts to add in this document are:
- extending Decision Tree experiments and model behavior analysis

So this document will progressively become the Decision Tree reference for the whole sub-phase, connecting splits, implemented reusable utilities, impurity, tree structure, and hierarchical rule-based prediction.