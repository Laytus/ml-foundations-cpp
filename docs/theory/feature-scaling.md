# Feature Scaling

## Purpose

The goal of this sub-phase is to understand how preprocessing changes optimization behavior in Machine Learning.

At this stage of the project, feature scaling is studied as a transformation applied to raw input data before training.

This sub-phase is important because many optimization methods are sensitive to the numerical scale of the input features.

So feature scaling is not only a data-cleaning topic. It is also an optimization topic.

---

## Why feature scaling matters

Different features can naturally live on very different numerical scales.

For example:
- one feature may vary between `0` and `1`
- another may vary between `1000` and `5000`

If features are not scaled, optimization can become harder to interpret and less efficient.

This matters especially for gradient-based models, because:
- large-scale features can dominate updates
- learning dynamics can become uneven
- convergence may become slower or less stable

So feature scaling helps make optimization more balanced and easier to study.

---

## Min-max scaling

### Purpose

Min-max scaling is a preprocessing technique that rescales numeric values into a fixed range.

In the first version of this project, the target range is:

$$
[0, 1]
$$

So the goal is to transform a feature so that:
- the minimum value becomes `0`
- the maximum value becomes `1`
- all other values are mapped proportionally between them

### Core idea

Given a value \(x\), and the minimum and maximum values of its feature:

$$
x_{\min}, \quad x_{\max}
$$

The min-max scaled version is:

$$
x_{\text{scaled}} = \frac{x - x_{\min}}{x_{\max} - x_{\min}}
$$

This maps the original feature into the range:

$$
[0, 1]
$$

### Intuition

This transformation does two things.

#### 1. Shift

Subtracting \(x_{\min}\) moves the minimum value to `0`.

#### 2. Normalize by range

Dividing by \(x_{\max} - x_{\min}\) compresses or stretches the values so that the maximum becomes `1`.

So min-max scaling is really:
- shift to start at zero
- divide by the feature range

### Example

Suppose the feature values are:

$$
[10, 15, 20]
$$

Then:
- \(x_{\min} = 10\)
- \(x_{\max} = 20\)
- range = \(20 - 10 = 10\)

Now scale each value.

#### For 10

$$
\frac{10 - 10}{20 - 10} = 0
$$

#### For 15

$$
\frac{15 - 10}{20 - 10} = \frac{5}{10} = 0.5
$$

#### For 20

$$
\frac{20 - 10}{20 - 10} = 1
$$

So the scaled values are:

$$
[0, 0.5, 1]
$$

### What min-max scaling preserves

Min-max scaling preserves:
- ordering
- relative position within the feature range

So if:
- \(a < b\) before scaling

then:
- scaled\((a) <\) scaled\((b)\) after scaling

This is important because the transformation changes scale, but not rank.

### What min-max scaling does not preserve

Min-max scaling does not preserve:
- original units
- original distances in absolute terms
- robustness to outliers

If one value is extremely large or small, it can stretch the whole range and compress most other points into a narrow interval.

So min-max scaling is simple and useful, but sensitive to extreme values.

### Why it matters for this project

In the current project, min-max scaling is important because it creates the first explicit bridge between:
- raw data
- preprocessing
- optimization behavior

It helps answer questions like:
- why does scaling change convergence speed?
- why do some learning rates work better after scaling?
- why should preprocessing be part of an ML pipeline?

So this is not only a data-transformation topic. It is also an optimization topic.

### Current implementation status

At the current stage of the project, the first Min-max scaling utility is implemented.

The current project supports:
- one-dimensional Min-max scaling through `ml::common::min_max_scale(...)`
- deterministic scaling into the range `[0, 1]`
- validation for empty input
- validation for zero-range input

So Min-max scaling is no longer only a theoretical preprocessing idea in the project. It is now also an implemented transformation utility.

### First scope in this project

For the first version, min-max scaling should stay simple:
- one-dimensional feature vector
- output range fixed to `[0, 1]`
- deterministic behavior
- clean validation
- no generic multi-feature matrix scaling yet

That is enough for a strong first implementation.

### Important edge case

If all values are equal, then:

$$
x_{\max} - x_{\min} = 0
$$

and the formula would divide by zero.

So the implementation must explicitly decide what to do in this case.

For a first clean version, the best choice is:
- throw an exception for zero-range input

because scaling is not meaningfully defined in the standard min-max formula when all values are identical.

### Current empirical observations

The current implementation already shows the expected behavior on simple test vectors.

#### Standard increasing example

For:

$$
[10, 15, 20]
$$

The output is:

$$
[0, 0.5, 1]
$$

#### Negative-to-positive example

For:

$$
[-10, 0, 10]
$$

The output is also:

$$
[0, 0.5, 1]
$$

This confirms that Min-max scaling depends on relative position within the range, not on the sign of the values.

#### Non-symmetric example

For:

$$
[2, 4, 8]
$$

The output is approximately:

$$
[0, 0.333333, 1]
$$

This confirms that the transformation is proportional, not based on equal spacing assumptions.

#### Constant-vector case

For a vector such as:

$$
[2, 2, 2]
$$

The implementation throws an exception because the range is zero.

#### Empty-vector case

For:

$$
[]
$$

The implementation also throws an exception because Min-max scaling is undefined for empty input.

These observations match the intended first-scope behavior of the project.

---

## Min-max scaling as a preprocessing utility

At this stage of the project, Min-max scaling is now represented at two levels:
- as a theoretical preprocessing concept
- as a concrete reusable utility in the common module

This is important because preprocessing should not remain only descriptive. It should become part of the actual ML workflow.

The current implementation makes it possible to:
- transform one-dimensional feature vectors before training
- test scaling behavior directly
- prepare later comparisons between raw vs scaled optimization behavior

This creates the first practical bridge between:
- preprocessing theory
- reusable project utilities
- optimization experiments

---

## Standardization

### Purpose

Standardization is a preprocessing technique that rescales a feature so that it is centered around zero and expressed in units of standard deviation.

In the first version of this project, the goal is to transform a feature so that:
- its mean becomes approximately `0`
- its scale is normalized using its standard deviation

Unlike Min-max scaling, standardization does not map values into a fixed interval such as `[0, 1]`.

### Core idea

Given a value \(x\), the feature mean \(\mu\), and the feature standard deviation \(\sigma\), the standardized version is:

$$
x_{\text{standardized}} = \frac{x - \mu}{\sigma}
$$

This means the transformation does two things:
- subtract the mean
- divide by the standard deviation

### Intuition

This transformation can be understood in two steps.

#### 1. Centering

Subtracting the mean moves the feature so that its center is at zero.

So values above the mean become positive, and values below the mean become negative.

#### 2. Scaling by spread

Dividing by the standard deviation expresses each value in units of the feature's spread.

So a standardized value tells how far the original value is from the mean relative to the typical scale of variation in the feature.

### Example

Suppose the feature values are:

$$
[10, 15, 20]
$$

Then the mean is:

$$
\mu = 15
$$

Using population standard deviation for the first version of the project:

$$
\sigma = \sqrt{\frac{(10-15)^2 + (15-15)^2 + (20-15)^2}{3}}
= \sqrt{\frac{25 + 0 + 25}{3}}
= \sqrt{\frac{50}{3}}
\approx 4.08248
$$

Now standardize each value.

#### For 10

$$
\frac{10 - 15}{4.08248} \approx -1.22474
$$

#### For 15

$$
\frac{15 - 15}{4.08248} = 0
$$

#### For 20

$$
\frac{20 - 15}{4.08248} \approx 1.22474
$$

So the standardized values are approximately:

$$
[-1.22474, 0, 1.22474]
$$

### What standardization preserves

Standardization preserves:
- ordering
- relative spacing structure up to a linear transformation
- whether values are above or below the mean

So if:
- \(a < b\) before standardization

then:
- standardized\((a) <\) standardized\((b)\) after standardization

### What standardization does not preserve

Standardization does not preserve:
- original units
- a fixed output interval
- robustness to outliers

Extreme values can strongly affect both the mean and the standard deviation.

So standardization is useful, but still sensitive to outliers.

### Why it matters for this project

In the current project, standardization is important because it provides a second major scaling strategy with a different optimization effect from Min-max scaling.

It helps answer questions like:
- why do some models work better with centered features?
- how does centering affect gradient-based optimization?
- why is scaling by spread different from scaling by min and max?

So standardization is not only a preprocessing technique. It is also part of understanding optimization behavior.

### Current implementation status

At the current stage of the project, the first standardization utility is implemented.

The current project supports:
- one-dimensional standardization through `ml::common::standardize(...)`
- deterministic transformation using population mean and population standard deviation
- validation for empty input
- validation for zero-standard-deviation input

So standardization is no longer only a theoretical preprocessing idea in the project. It is now also an implemented transformation utility.

### Relationship with Min-max scaling

Both Min-max scaling and standardization are linear feature transformations, but they emphasize different properties.

#### Min-max scaling
- maps values into `[0, 1]`
- depends on minimum and maximum
- preserves relative position in the observed range

#### Standardization
- centers values around `0`
- scales by standard deviation
- expresses values relative to the feature's spread

So the two methods may produce different optimization behavior even on the same dataset.

### First scope in this project

For the first version, standardization should stay simple:
- one-dimensional feature vector
- population mean
- population standard deviation
- deterministic behavior
- clean validation
- no multi-feature matrix standardization yet

That is enough for a strong first implementation.

### Important edge case

If all values are equal, then the standard deviation is zero:

$$
\sigma = 0
$$

and the formula would divide by zero.

So the implementation must explicitly decide what to do in this case.

For a first clean version, the best choice is:
- throw an exception for zero-standard-deviation input

because standardization is not meaningfully defined in the standard form when the feature has no variation.

### Current empirical observations

The current implementation already shows the expected behavior on simple test vectors.

#### Canonical three-value example

For:

$$
[10, 15, 20]
$$

The output is approximately:

$$
[-1.22474, 0, 1.22474]
$$

#### Centered symmetric example

For:

$$
[-1, 0, 1]
$$

The output is also approximately:

$$
[-1.22474, 0, 1.22474]
$$

This confirms that standardization depends on centering and spread, not on the original numeric offset.

#### Another equally spaced example

For:

$$
[2, 4, 6]
$$

The output is again approximately:

$$
[-1.22474, 0, 1.22474]
$$

This confirms that equally spaced three-value vectors can collapse to the same standardized pattern.

#### Non-symmetric example

For:

$$
[1, 2, 10]
$$

The output is approximately:

$$
[-0.827606, -0.579324, 1.40693]
$$

This confirms that standardization reflects distance from the mean relative to the feature's spread.

#### Empty-vector case

For:

$$
[]
$$

The implementation throws an exception because standardization is undefined for empty input.

#### Constant-vector case

For:

$$
[5, 5, 5]
$$

The implementation throws an exception because the standard deviation is zero.

These observations match the intended first-scope behavior of the project.

---

## Standardization as a preprocessing utility

At this stage of the project, standardization is now represented at two levels:
- as a theoretical preprocessing concept
- as a concrete reusable utility in the common module

This is important because centered and scale-normalized features can influence optimization differently from Min-max scaled features.

The current implementation makes it possible to:
- transform one-dimensional feature vectors before training
- test centering and scaling behavior directly
- prepare later comparisons between Min-max scaling and standardization
- prepare later comparisons between unscaled vs scaled optimization behavior

This creates the second practical bridge between:
- preprocessing theory
- reusable project utilities
- optimization experiments

---

## Min-max scaling vs standardization

### Why this comparison matters

At this stage of the project, both Min-max scaling and standardization are implemented.

So it is now important to compare them explicitly, not only as formulas, but as preprocessing choices that can affect optimization differently.

This comparison matters because two scaling methods can both be valid while still producing different training behavior.

### Shared properties

Both methods:
- are linear feature transformations
- preserve ordering
- remove the original units of the feature
- can change optimization behavior
- are sensitive to outliers

So they both belong to the same general family of preprocessing tools.

### Main difference in objective

The two methods normalize a feature in different ways.

#### Min-max scaling

Min-max scaling uses:
- the minimum value
- the maximum value

Its goal is to place the feature inside a fixed interval:

$$
[0, 1]
$$

So it emphasizes position within the observed range.

#### Standardization

Standardization uses:
- the mean
- the standard deviation

Its goal is to center the feature around zero and express values relative to the feature's spread.

So it emphasizes distance from the mean in standardized units.

### Output behavior

#### Min-max scaling

Typical output properties:
- minimum becomes `0`
- maximum becomes `1`
- all values lie in `[0, 1]`

#### Standardization

Typical output properties:
- mean becomes approximately `0`
- population standard deviation becomes approximately `1`
- values are not constrained to a fixed interval

So Min-max scaling controls the range, while standardization controls centering and spread.

### Interpretation difference

#### Min-max scaling

A scaled value tells where the original value lies between the observed minimum and maximum.

For example:
- `0` means the minimum
- `1` means the maximum
- `0.5` means the midpoint of the observed range

#### Standardization

A standardized value tells how far the original value is from the mean in units of standard deviation.

For example:
- `0` means exactly at the mean
- positive values mean above the mean
- negative values mean below the mean
- larger magnitude means farther from the mean relative to the feature's spread

### Sensitivity difference

Both methods are sensitive to outliers, but in different ways.

#### Min-max scaling

An extreme minimum or maximum can stretch the whole range and compress most values into a narrow interval.

#### Standardization

Extreme values can strongly affect both the mean and the standard deviation, changing the centered and scaled output for all points.

So both methods are sensitive to extreme values, but through different summary statistics.

### Optimization intuition

#### Min-max scaling

Min-max scaling can be useful when it is desirable to keep all values inside a predictable range.

This can make the scale of the feature easier to inspect directly.

#### Standardization

Standardization can be useful when centering around zero helps make optimization more balanced.

This is especially relevant in gradient-based settings, where centered features often interact differently with parameter updates.

So the two methods may influence optimization differently even when applied to the same raw feature.

### First practical comparison in this project

For the current project, the first useful comparison should remain simple.

The main questions are:
- how do the transformed values differ on the same vector?
- how does each method change the numerical scale of the feature?
- how does each method later affect optimization experiments?

The first goal is not to decide that one method is universally better.

The goal is to understand that different preprocessing choices create different optimization conditions.

### Current project status

At the current stage of the project:
- Min-max scaling is implemented through `ml::common::min_max_scale(...)`
- standardization is implemented through `ml::common::standardize(...)`

So this comparison is no longer only theoretical. It can now be explored directly in code and later tested inside model-training workflows.

---

## Observing the effect of scaling on optimization

### Why this matters

At this stage of the project, feature scaling is no longer only theoretical.

Both Min-max scaling and standardization are now implemented, so it becomes possible to study how scaling changes optimization behavior in practice.

This matters because preprocessing does not only change the numerical appearance of the data. It can also change how efficiently and how stably an optimizer behaves.

So the next important question is not only:
- what transformation was applied?

but also:
- how did that transformation affect training?

### Current implementation status

At the current stage of the project, the first direct scaling-vs-optimization comparison workflow is implemented.

The current project now has:
- raw-feature training
- Min-max scaled training
- standardized-feature training
- Batch GD experiment execution
- stability analysis
- convergence analysis

So the effect of scaling on optimization is no longer only theoretical in the project. It can now be observed directly through controlled training comparisons.

### Core idea

The same model trained on:
- raw features
- Min-max scaled features
- standardized features

may show different optimization behavior even when:
- the dataset is the same
- the target values are the same
- the learning rate is the same
- the number of epochs is the same

So scaling should be studied as something that changes the optimization conditions of the problem.

### What may change after scaling

When feature scaling is introduced, several optimization properties may change.

#### 1. Convergence speed

The model may approach a good parameter region faster or slower depending on the feature scale.

#### 2. Stability

Some learning rates that behave poorly on raw features may behave better after scaling.

#### 3. Gradient magnitude

Because the feature values enter the gradient computations directly, changing feature scale can change update magnitudes.

#### 4. Interpretability of training behavior

Scaled features can make optimization trajectories easier to compare and reason about.

So scaling can influence not only performance, but also how understandable training dynamics become.

### Why this is especially important for gradient-based models

In gradient-based models such as Linear Regression and later Logistic Regression, the feature values directly affect the loss gradients.

So if one feature has a much larger numerical scale, parameter updates can become uneven or harder to control.

Feature scaling helps reduce this imbalance.

That is why scaling and optimization should not be studied separately.

### Raw vs scaled comparison in this project

For the current project, the first useful comparison should remain simple.

The main idea is to compare the same training setup under different preprocessing choices:
- no scaling
- Min-max scaling
- standardization

The goal is not yet to make broad theoretical claims.

The goal is to observe directly:
- whether MSE decreases faster
- whether convergence classifications change
- whether stability classifications change
- whether learning-rate sensitivity changes

### What this comparison can teach

Studying the effect of scaling on optimization helps build intuition about:
- why preprocessing matters before training
- why optimization is sensitive to numerical scale
- why the same optimizer can behave differently on transformed inputs
- why ML workflows should include both preprocessing and optimization analysis

This makes feature scaling a genuine part of model training, not only a preprocessing footnote.

### First scope in this project

For the first version, the effect-of-scaling analysis should stay simple:
- one-dimensional datasets
- one model at a time
- one optimizer setup at a time
- direct comparison between raw, Min-max scaled, and standardized features

That is enough to build strong practical intuition.

### Current project status

At the current stage of the project:
- raw-feature training already exists
- Min-max scaling is implemented
- standardization is implemented
- optimization experiments already exist for Batch GD and SGD
- stability and convergence analysis already exist

So the project now has the pieces needed to study the effect of scaling on optimization directly.

### Current empirical observations

The current project already shows a strong difference between raw and scaled training behavior under the same optimization setup.

Using the same:
- dataset
- target values
- model initialization
- Batch GD optimizer
- learning rate
- epoch count

raw features, Min-max scaled features, and standardized features can lead to very different outcomes.

#### Raw-feature case

For a representative experiment on raw features:
- training diverges
- MSE explodes rapidly
- stability is classified as `Divergent`
- convergence is classified as `Failed`

This shows that the unscaled feature magnitude can make the chosen learning rate unusable.

#### Min-max scaled case

For the same optimization setup applied to Min-max scaled features:
- training remains bounded
- MSE decreases strongly
- stability is classified as `Stable`
- convergence is classified as `Strong`

This shows that Min-max scaling can turn an unstable training setup into a usable one.

#### Standardized case

For the same optimization setup applied to standardized features:
- training remains bounded
- MSE decreases even more strongly
- stability is classified as `Stable`
- convergence is classified as `Strong`

In the current experiment, standardization produces the best final loss among the tested variants.

### First conclusion from the current project

The current empirical comparison supports three important conclusions.

#### 1. Scaling can determine whether optimization succeeds at all

With the same optimizer settings:
- raw training can diverge
- scaled training can converge

So scaling is not only a numerical convenience. It can completely change whether training works.

#### 2. Different scaling methods can both help, but not equally

Both Min-max scaling and standardization stabilize training in the current project, but they do not produce exactly the same optimization outcome.

So preprocessing choice still matters even after deciding to scale the data.

#### 3. Standardization appears especially effective in the current Batch GD comparison

In the current experiment, standardization produces the strongest convergence result among the tested variants.

This is not yet a universal claim, but it is an important empirical observation for the current project.

---

## What comes next in this doc

The next concepts to add in this document are:
- extending scaling comparisons to additional training experiments

So this document will progressively become the feature-scaling reference for the whole sub-phase, connecting preprocessing theory, implementation status, transformation utilities, controlled training comparisons, and optimization effects.