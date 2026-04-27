# Evaluation Metrics

## Purpose

The goal of this sub-phase is to understand how classification models should be evaluated.

At this stage of the project, Logistic Regression can already:
- produce probability-like outputs
- produce binary class predictions
- learn through batch training

So the next question is not how the model learns, but how to judge whether its predictions are good.

This sub-phase is important because a trainable classifier is not complete without a way to evaluate its performance.

---

## Why evaluation metrics matter

A classification model can make predictions, but those predictions need to be measured in a structured way.

Without evaluation metrics, it is difficult to answer questions such as:
- how often is the model correct?
- how often does it fail?
- which types of mistakes does it make?

So evaluation metrics are the bridge between:
- model predictions
- model performance
- model comparison

They turn raw outputs into interpretable performance summaries.

---

## Accuracy

### Purpose

Accuracy is the first and simplest classification metric in the project.

Its role is to measure how often the predicted class matches the true class.

For a first evaluation metric, it is a natural starting point because:
- it is simple
- it is intuitive
- it connects directly to binary class predictions

### Definition

For a dataset with \(n\) samples, accuracy is:

$$
\text{Accuracy} = \frac{\text{number of correct predictions}}{n}
$$

Equivalently, if:
- \(y_i\) is the true label
- \(\hat{y}_i\) is the predicted class label

then:

$$
\text{Accuracy} = \frac{1}{n}\sum_{i=0}^{n-1} \mathbf{1}(\hat{y}_i = y_i)
$$

where \(\mathbf{1}(\hat{y}_i = y_i)\) is `1` when the prediction is correct and `0` otherwise.

So accuracy is simply the fraction of correct predictions.

### Intuition

Accuracy answers the question:

> Out of all predictions, how many were correct?

If a model predicts all classes perfectly, the accuracy is:

$$
1
$$

If the model gets half of them correct, the accuracy is:

$$
0.5
$$

So accuracy always lies in the interval:

$$
0 \le \text{Accuracy} \le 1
$$

Sometimes it is also expressed as a percentage.

### Example

Suppose the true labels are:

$$
[0, 1, 1, 0]
$$

and the predicted labels are:

$$
[0, 1, 0, 0]
$$

Compare them sample by sample:
- first prediction: correct
- second prediction: correct
- third prediction: wrong
- fourth prediction: correct

So there are `3` correct predictions out of `4` total.

Therefore:

$$
\text{Accuracy} = \frac{3}{4} = 0.75
$$

So the model is correct `75%` of the time on this dataset.

### Why accuracy is useful

Accuracy is useful because it gives a fast global summary of classifier performance.

It is especially useful in early-stage testing because:
- it is easy to compute
- it is easy to interpret
- it quickly shows whether the model is learning something meaningful

For the current project, this makes it a very good first evaluation metric.

### Limitation of accuracy

Although accuracy is useful, it does not explain which types of mistakes the model makes.

Two models can have the same accuracy while making different error patterns.

So accuracy is a good starting point, but not a complete description of classifier behavior.

This is why later metrics such as precision, recall, and confusion matrix will still be needed.

### Why this matters for the project

In the current project, accuracy is important because it creates the first evaluation bridge between:
- Logistic Regression predictions
- binary labels
- measurable classifier performance

Without accuracy, the model can predict classes, but there is not yet a standard summary of how well it performs.

With accuracy, the project gains its first direct classification metric.

### Current implementation status

At the current stage of the project, the first evaluation metric utility is implemented.

The current project supports:
- dataset-level accuracy through `ml::common::accuracy(const std::vector<int>& y_true, const std::vector<int>& y_pred)`
- validation for empty vectors
- validation for size mismatch
- validation for invalid binary labels in both true and predicted vectors

So accuracy is no longer only a theoretical evaluation metric in the project. It is now also an implemented evaluation utility.

### First scope in this project

For the first version, accuracy should stay simple:
- binary class labels only
- predicted class labels only
- dataset-level accuracy only
- no weighted variants
- no multi-class support yet

That is enough for a strong first implementation.

### Current empirical observations

The current implementation already shows the expected behavior on simple binary examples.

#### Perfect accuracy case

For:

$$
y_{\text{true}} = [0, 1, 1, 0]
$$

and:

$$
y_{\text{pred}} = [0, 1, 1, 0]
$$

the observed accuracy is:

$$
1.0
$$

This matches the expected result for perfect prediction.

#### Partial accuracy case

For:

$$
y_{\text{true}} = [0, 1, 1, 0]
$$

and:

$$
y_{\text{pred}} = [0, 1, 0, 0]
$$

the observed accuracy is:

$$
0.75
$$

This matches the expected result of `3` correct predictions out of `4`.

#### Zero-accuracy case

For:

$$
y_{\text{true}} = [0, 1]
$$

and:

$$
y_{\text{pred}} = [1, 0]
$$

the observed accuracy is:

$$
0.0
$$

This matches the expected result when every prediction is wrong.

#### Size-mismatch case

When the true-label and predicted-label vectors have different sizes, the implementation throws an exception.

#### Empty-vector case

When the input vectors are empty, the implementation throws an exception.

#### Invalid-label cases

When either vector contains a label outside the binary set `{0, 1}`, the implementation throws an exception.

These observations match the intended first-scope behavior of the project.

### Important practical note

Accuracy should be computed from final class predictions, not from raw probabilities.

That means the typical workflow is:
1. compute class predictions
2. compare them with the true labels
3. count how many match
4. divide by the total number of samples

So accuracy belongs after the prediction step, not before it.

---

## Accuracy as an evaluation utility

At this stage of the project, accuracy is now represented at two levels:
- as a theoretical classification metric
- as a concrete reusable evaluation utility in the common module

This is important because the project now has:
- a trainable binary classifier
- final class predictions
- a first dataset-level performance summary

So the project now has the essential first evaluation layer needed before moving to more detailed metrics such as precision, recall, and confusion matrix.

---

## Precision and Recall

### Purpose

Once accuracy is available, the next step is to measure classifier performance in a more detailed way.

Accuracy tells how often the model is correct overall, but it does not distinguish between different types of classification errors.

Precision and recall provide that extra detail.

They are especially important in binary classification because they separate:
- mistakes made when predicting the positive class
- mistakes made by failing to recover the positive class

So precision and recall deepen the evaluation layer of the project.

### Why accuracy is not enough

A classifier may have high accuracy and still behave poorly on one class.

For example:
- it may predict the positive class too often
- it may almost never predict the positive class
- it may miss many true positives
- it may produce many false alarms

Accuracy alone cannot show these behaviors clearly.

So precision and recall are needed to understand *how* the model is right or wrong, not only *how often* it is right.

### Basic class outcomes

To define precision and recall, four basic outcomes are used.

#### True Positive (TP)

The model predicts class `1`, and the true label is also `1`.

#### False Positive (FP)

The model predicts class `1`, but the true label is `0`.

#### False Negative (FN)

The model predicts class `0`, but the true label is `1`.

#### True Negative (TN)

The model predicts class `0`, and the true label is also `0`.

Precision and recall are built from these quantities.

### Precision

#### Definition

Precision measures how reliable the positive predictions are.

It answers the question:

> Out of all samples predicted as positive, how many were truly positive?

The formula is:

$$
\text{Precision} = \frac{TP}{TP + FP}
$$

So precision focuses on the quality of the predicted positive class.

#### Intuition

If precision is high, then when the model says “positive,” it is usually correct.

If precision is low, then many positive predictions are actually wrong.

So precision penalizes false positives.

#### Example

Suppose:
- \(TP = 8\)
- \(FP = 2\)

Then:

$$
\text{Precision} = \frac{8}{8+2} = \frac{8}{10} = 0.8
$$

So `80%` of positive predictions are correct.

### Recall

#### Definition

Recall measures how well the model recovers the true positive class.

It answers the question:

> Out of all truly positive samples, how many did the model correctly detect?

The formula is:

$$
\text{Recall} = \frac{TP}{TP + FN}
$$

So recall focuses on how completely the model captures the positive class.

#### Intuition

If recall is high, then the model detects most of the true positives.

If recall is low, then the model misses many true positives.

So recall penalizes false negatives.

#### Example

Suppose:
- \(TP = 8\)
- \(FN = 4\)

Then:

$$
\text{Recall} = \frac{8}{8+4} = \frac{8}{12} \approx 0.6667
$$

So the model detects about `66.67%` of the true positives.

### Precision vs Recall

Precision and recall measure different aspects of classifier behavior.

#### Precision asks:
- when the model predicts positive, is it usually correct?

#### Recall asks:
- when a sample is truly positive, does the model usually find it?

So:
- precision is about prediction reliability
- recall is about positive-class coverage

A model can have:
- high precision and low recall
- low precision and high recall
- high values for both
- low values for both

That is why both metrics are needed.

### Example comparison

Suppose two models both have the same accuracy, but:

- Model A predicts positive only when it is very sure
- Model B predicts positive more often

Then:
- Model A may have higher precision
- Model B may have higher recall

So precision and recall reveal behavior that accuracy alone hides.

### Why this matters for the project

In the current project, precision and recall are important because they expand the evaluation layer beyond simple overall correctness.

They create the bridge between:
- binary predictions
- types of classification mistakes
- more detailed model evaluation

Without them, the project can only say how often the classifier is right overall.

With them, the project can describe *how* the classifier behaves on the positive class.

### Current implementation status

At the current stage of the project, the precision and recall utilities are implemented.

The current project supports:
- dataset-level precision through `ml::common::precision(const std::vector<int>& y_true, const std::vector<int>& y_pred)`
- dataset-level recall through `ml::common::recall(const std::vector<int>& y_true, const std::vector<int>& y_pred)`
- validation for empty vectors
- validation for size mismatch
- validation for invalid binary labels in both true and predicted vectors
- a first explicit zero-denominator policy returning `0.0`

So precision and recall are no longer only theoretical evaluation metrics in the project. They are now also implemented evaluation utilities.

### First scope in this project

For the first version, precision and recall should stay simple:
- binary classification only
- dataset-level metrics only
- computed from final class predictions
- no averaging across classes
- no multi-class support yet

That is enough for a strong first implementation.

### Current empirical observations

The current implementation already shows the expected behavior on simple binary examples.

#### Perfect prediction case

For:

$$
y_{\text{true}} = [0, 1, 1, 0]
$$

and:

$$
y_{\text{pred}} = [0, 1, 1, 0]
$$

the observed values are:

$$
\text{Precision} = 1.0
$$

$$
\text{Recall} = 1.0
$$

This matches the expected result for perfect classification.

#### False-positive case

For:

$$
y_{\text{true}} = [0, 1, 1, 0]
$$

and:

$$
y_{\text{pred}} = [1, 1, 1, 0]
$$

the observed values are:

$$
\text{Precision} \approx 0.666667
$$

$$
\text{Recall} = 1.0
$$

This matches the expected behavior:
- precision decreases because there is one false positive
- recall remains perfect because all true positives are still recovered

#### False-negative case

For:

$$
y_{\text{true}} = [0, 1, 1, 0]
$$

and:

$$
y_{\text{pred}} = [0, 1, 0, 0]
$$

the observed values are:

$$
\text{Precision} = 1.0
$$

$$
\text{Recall} = 0.5
$$

This matches the expected behavior:
- precision stays perfect because the model makes no false positive predictions
- recall decreases because one true positive is missed

#### No predicted positives case

For:

$$
y_{\text{true}} = [0, 1, 1, 0]
$$

and:

$$
y_{\text{pred}} = [0, 0, 0, 0]
$$

the observed values are:

$$
\text{Precision} = 0.0
$$

$$
\text{Recall} = 0.0
$$

This matches the current zero-denominator policy and the fact that the model detects no positive samples.

#### No true positives in the dataset case

For:

$$
y_{\text{true}} = [0, 0, 0, 0]
$$

and:

$$
y_{\text{pred}} = [0, 1, 0, 1]
$$

the observed values are:

$$
\text{Precision} = 0.0
$$

$$
\text{Recall} = 0.0
$$

This matches the current implementation policy for the first project version.

#### Size-mismatch case

When the true-label and predicted-label vectors have different sizes, the implementation throws an exception.

#### Empty-vector case

When the input vectors are empty, the implementation throws an exception.

#### Invalid-label cases

When either vector contains a label outside the binary set `{0, 1}`, the implementation throws an exception.

These observations match the intended first-scope behavior of the project.

### Important practical note

Precision becomes undefined when:

$$
TP + FP = 0
$$

because the model never predicts the positive class.

Recall becomes undefined when:

$$
TP + FN = 0
$$

because there are no true positive samples in the dataset.

So the implementation will need to make an explicit decision about how to handle these edge cases.

---

## Precision and Recall as evaluation utilities

At this stage of the project, precision and recall are now represented at two levels:
- as theoretical classification metrics
- as concrete reusable evaluation utilities in the common module

This is important because the project now has:
- overall correctness through accuracy
- positive-prediction reliability through precision
- positive-class coverage through recall

So the project now has a more detailed evaluation layer that can distinguish between false-positive-heavy and false-negative-heavy behavior.

---

## Confusion Matrix

### Purpose

Once accuracy, precision, and recall are available, the next step is to organize classification outcomes in a single structured view.

The confusion matrix provides that structure.

Its role is to summarize how predictions and true labels interact across the two classes.

So instead of looking at each metric separately, the confusion matrix shows the full error pattern of the classifier in one place.

### Core idea

For binary classification, every prediction falls into one of four categories:
- true positive
- false positive
- false negative
- true negative

A confusion matrix arranges those counts into a table.

For the first version of the project, the binary confusion matrix should be understood as:

$$
\begin{array}{c|cc}
 & \text{Predicted 0} & \text{Predicted 1} \\
\hline
\text{True 0} & TN & FP \\
\text{True 1} & FN & TP
\end{array}
$$

This means the confusion matrix is not a new kind of metric by itself.

It is a structured summary of the classifier outcomes from which several metrics can be derived.

### Interpretation of the four cells

#### True Negative (TN)

The model predicts class `0`, and the true label is `0`.

#### False Positive (FP)

The model predicts class `1`, but the true label is `0`.

This is also called a false alarm.

#### False Negative (FN)

The model predicts class `0`, but the true label is `1`.

This is also called a missed positive.

#### True Positive (TP)

The model predicts class `1`, and the true label is `1`.

Together, these four values describe all classifier outcomes in binary classification.

### Example

Suppose:

$$
y_{\text{true}} = [0, 1, 1, 0]
$$

and:

$$
y_{\text{pred}} = [0, 1, 0, 0]
$$

Compare them sample by sample:
- first sample: true `0`, predicted `0` → `TN`
- second sample: true `1`, predicted `1` → `TP`
- third sample: true `1`, predicted `0` → `FN`
- fourth sample: true `0`, predicted `0` → `TN`

So the counts are:
- `TN = 2`
- `FP = 0`
- `FN = 1`
- `TP = 1`

The confusion matrix becomes:

$$
\begin{array}{c|cc}
 & \text{Predicted 0} & \text{Predicted 1} \\
\hline
\text{True 0} & 2 & 0 \\
\text{True 1} & 1 & 1
\end{array}
$$

### Why the confusion matrix is useful

The confusion matrix is useful because it preserves more information than a single scalar metric.

For example:
- accuracy compresses everything into one number
- precision focuses on positive prediction reliability
- recall focuses on positive class coverage

The confusion matrix keeps all four outcome counts visible.

So it allows the project to:
- inspect error patterns directly
- explain why precision has a certain value
- explain why recall has a certain value
- connect all classification metrics back to the same base counts

### Relationship with previous metrics

The confusion matrix is strongly connected to the metrics already introduced.

From the confusion matrix:

$$
\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}
$$

$$
\text{Precision} = \frac{TP}{TP + FP}
$$

$$
\text{Recall} = \frac{TP}{TP + FN}
$$

So the confusion matrix acts as a foundational representation for binary classification evaluation.

### Why this matters for the project

In the current project, the confusion matrix is important because it completes the first evaluation layer for binary classification.

It creates the bridge between:
- predicted class labels
- true class labels
- structured error analysis
- previously implemented scalar metrics

Without the confusion matrix, the project has evaluation metrics, but not yet a compact structural summary of the classifier behavior.

With it, the project can describe not only how well the classifier performs, but also what kinds of mistakes it makes.

### Current implementation status

At the current stage of the project, the confusion-matrix utility is implemented.

The current project supports:
- dataset-level confusion matrix computation through `ml::common::confusion_matrix(const std::vector<int>& y_true, const std::vector<int>& y_pred)`
- a reusable `ConfusionMatrix` structure containing:
  - `true_negatives`
  - `false_positives`
  - `false_negatives`
  - `true_positives`
- validation for empty vectors
- validation for size mismatch
- validation for invalid binary labels in both true and predicted vectors

So the confusion matrix is no longer only a theoretical evaluation concept in the project. It is now also an implemented evaluation utility.

### First scope in this project

For the first version, the confusion matrix should stay simple:
- binary classification only
- dataset-level counts only
- no normalized variants
- no multi-class support yet
- direct counts for `TN`, `FP`, `FN`, and `TP`

That is enough for a strong first implementation.

### Current empirical observations

The current implementation already shows the expected behavior on simple binary examples.

#### Mixed example

For:

$$
y_{\text{true}} = [0, 1, 1, 0]
$$

and:

$$
y_{\text{pred}} = [0, 1, 0, 0]
$$

the observed confusion matrix is:

$$
TN = 2, \quad FP = 0, \quad FN = 1, \quad TP = 1
$$

This matches the expected result from the worked example in the theory section.

#### Perfect-prediction case

For:

$$
y_{\text{true}} = [0, 1, 1, 0]
$$

and:

$$
y_{\text{pred}} = [0, 1, 1, 0]
$$

the observed confusion matrix is:

$$
TN = 2, \quad FP = 0, \quad FN = 0, \quad TP = 2
$$

This matches the expected result for perfect classification.

#### All-wrong case

For:

$$
y_{\text{true}} = [0, 1]
$$

and:

$$
y_{\text{pred}} = [1, 0]
$$

the observed confusion matrix is:

$$
TN = 0, \quad FP = 1, \quad FN = 1, \quad TP = 0
$$

This matches the expected result when every prediction is wrong.

#### Only-negative-predictions case

For:

$$
y_{\text{true}} = [0, 1, 1, 0]
$$

and:

$$
y_{\text{pred}} = [0, 0, 0, 0]
$$

the observed confusion matrix is:

$$
TN = 2, \quad FP = 0, \quad FN = 2, \quad TP = 0
$$

This matches the expected behavior when the model never predicts the positive class.

#### Only-positive-predictions case

For:

$$
y_{\text{true}} = [0, 1, 1, 0]
$$

and:

$$
y_{\text{pred}} = [1, 1, 1, 1]
$$

the observed confusion matrix is:

$$
TN = 0, \quad FP = 2, \quad FN = 0, \quad TP = 2
$$

This matches the expected behavior when the model always predicts the positive class.

#### Size-mismatch case

When the true-label and predicted-label vectors have different sizes, the implementation throws an exception.

#### Empty-vector case

When the input vectors are empty, the implementation throws an exception.

#### Invalid-label cases

When either vector contains a label outside the binary set `{0, 1}`, the implementation throws an exception.

These observations match the intended first-scope behavior of the project.

### Important practical note

The confusion matrix should be computed from final predicted class labels, not from raw probabilities.

So the workflow is:
1. compute final class predictions
2. compare them with true labels
3. count `TN`, `FP`, `FN`, and `TP`
4. organize those counts into a reusable representation

So, just like accuracy, precision, and recall, the confusion matrix belongs after the prediction step.

---

## Confusion Matrix as an evaluation utility

At this stage of the project, the confusion matrix is now represented at two levels:
- as a theoretical structural summary of binary classification outcomes
- as a concrete reusable evaluation utility in the common module

This is important because the project now has:
- overall correctness through accuracy
- positive-prediction reliability through precision
- positive-class coverage through recall
- full outcome counts through the confusion matrix

So the project now has a complete first binary-classification evaluation layer that can summarize and explain classifier behavior in a structured way.

---

## What comes next in this doc

The next concepts to add in this document are:
- extending evaluation experiments and model comparison

So this document will progressively become the evaluation reference for the whole sub-phase, connecting classifier outputs, performance summaries, implemented evaluation utilities, structured error analysis, and later model comparison.