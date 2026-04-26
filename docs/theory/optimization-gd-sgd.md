# Optimization (GD / SGD)

## Purpose

The goal of this sub-phase is to build a deep understanding of optimization in Machine Learning, starting from Gradient Descent and later extending to Stochastic Gradient Descent.

At this stage of the project, optimization is studied as the process that allows a trainable model to improve its parameters through repeated updates.

This sub-phase is important because many ML models are not only defined by:
- a prediction rule
- a loss function

They are also defined by how their parameters are optimized.

---

## What optimization means in ML

In this project, optimization means:

> finding better parameter values that reduce the model's loss.

A model such as Linear Regression has parameters:
- weight
- bias

These parameters determine the predictions of the model.

The loss function measures how wrong those predictions are.

Optimization is the process of changing the parameters so that the loss becomes smaller.

So the full trainable ML chain is:
1. make predictions
2. measure error with a loss function
3. compute how the loss changes with respect to the parameters
4. update the parameters to reduce the loss
5. repeat

This sub-phase focuses on steps 3–5.

---

## Why optimization is important

A model can have a correct mathematical form and still be useless if its parameters are poor.

For example, in Linear Regression:

$$
\hat{y} = wx + b
$$

The model structure may be correct, but if \(w\) and \(b\) are badly chosen, predictions will still be bad.

So optimization is what turns a parameterized model into a trainable model.

It is the mechanism that connects:
- theory
- training
- learning dynamics
- convergence behavior

---

## Gradient Descent (GD)

### Core idea

Gradient Descent is an optimization method that improves model parameters by moving them in the direction that reduces the loss.

The gradient tells us the direction of steepest increase of the loss.

So to reduce the loss, Gradient Descent moves the parameters in the opposite direction of the gradient.

### Update rule

For a parameter \(\theta\):

$$
\theta \leftarrow \theta - \alpha \frac{\partial L}{\partial \theta}
$$

Where:
- \(\theta\) = a model parameter
- \(L\) = loss function
- \(\alpha\) = learning rate

For Linear Regression with parameters \(w\) and \(b\):

$$
w \leftarrow w - \alpha \frac{\partial \mathrm{MSE}}{\partial w}
$$

$$
b \leftarrow b - \alpha \frac{\partial \mathrm{MSE}}{\partial b}
$$

### Intuition

The gradient gives a local direction for reducing the loss.

Each update is only one correction.

So learning is not a single jump, but a sequence of repeated corrections.

That is why Gradient Descent is inherently iterative.

---

## Batch Gradient Descent

### Definition

Batch Gradient Descent computes the gradient using the full dataset before each update.

So if the training set has \(n\) samples, every update step uses all \(n\) samples.

### Core idea

For each epoch:
1. use the full dataset
2. compute one average gradient
3. update the parameters once

So batch GD produces one update per full pass over the dataset.

### Why it is important

Batch GD is the cleanest first optimization method to study because:
- it is deterministic
- it is mathematically easier to analyze
- its convergence behavior is easier to observe
- it introduces less noise than SGD

For these reasons, it is the right first optimization method for this project.

### Relationship with the current project

In the current implementation of Linear Regression:
- the gradient is computed from the full dataset
- `gradient_descent_step(...)` performs one update
- `fit_batch(...)` repeats that update across epochs

So the current training mechanism already corresponds to Batch Gradient Descent.

This sub-phase now studies that mechanism explicitly as an optimization topic.

### Current implementation status

At the current stage of the project, Batch GD is already implemented and can now be studied explicitly as an optimization process.

The current project supports:
- one GD update through `gradient_descent_step(...)`
- repeated batch training through `fit_batch(...)`
- per-epoch history recording through `fit_batch_with_history(...)`
- explicit experiment packaging through `run_batch_gd_experiment(...)`

This means Batch GD is no longer only a training mechanism in the project. It is now also an object of structured observation and comparison.

---

## Learning rate

### Definition

The learning rate controls the size of each parameter update.

It appears in the update rule as \(\alpha\).

### Intuition

The learning rate determines how aggressively the model reacts to the gradient.

- small learning rate → cautious but slow updates
- large learning rate → faster but riskier updates

### Typical behaviors

#### If the learning rate is too small
- training improves slowly
- many epochs are required
- convergence may be very slow

#### If the learning rate is reasonable
- training improves efficiently
- loss decreases clearly
- parameters move toward a stable region

#### If the learning rate is too large
- updates may overshoot
- loss may oscillate
- training may diverge

This makes the learning rate one of the most important factors in optimization.

### Current empirical observations

The current implementation already shows the main qualitative learning-rate behaviors expected from Batch GD.

#### Small learning rate

With a small learning rate such as `0.01`:
- MSE decreases steadily
- parameter updates are smaller
- convergence is slower

#### Reasonable learning rate

With a learning rate such as `0.1`:
- MSE decreases very strongly
- parameters move quickly toward a good solution region
- training remains stable

#### Large learning rate

With a large learning rate such as `1.0`:
- MSE explodes instead of decreasing
- parameters overshoot heavily
- optimization diverges

These empirical observations match the expected theoretical behaviors of Batch GD.

---

## Optimization as a dynamic process

A very important idea in this sub-phase is that optimization is not only about a formula.

It is about a process over time.

What matters is not only one update, but also:
- how the loss evolves across epochs
- how the parameters evolve across epochs
- whether training stabilizes or diverges
- how hyperparameters such as the learning rate affect behavior

So optimization must be studied dynamically, not only statically.

This is why convergence analysis and stability analysis are natural parts of this sub-phase.

---

## Batch GD as an optimization experiment

At this stage of the project, Batch GD can now be run as an explicit optimization experiment.

One experiment contains:
- a chosen learning rate
- a chosen number of epochs
- the full training history

This is important because optimization should not only be executed. It should also be inspectable and comparable.

The current experiment-level view makes it possible to study:
- how MSE changes across epochs
- how `weight` changes across epochs
- how `bias` changes across epochs
- how learning rate changes training behavior

So Batch GD is now represented at two levels in the project:
- as a training mechanism
- as an optimization experiment

This separation is useful because it prepares the ground for later comparison with SGD.

---

## Stochastic Gradient Descent (SGD)

### Definition

Stochastic Gradient Descent updates the model parameters using only one training sample at a time.

So instead of computing the gradient from the full dataset before each update, SGD computes an update from a single sample.

If one sample is written as:
- \((x_i, y_i)\)

then one SGD update is based only on that sample.

### Core idea

For each sample:
1. compute the prediction for that sample
2. compute the loss contribution for that sample
3. compute the gradient contribution for that sample
4. update the parameters immediately

So SGD performs many smaller updates instead of one full-dataset update.

### Why SGD is important

SGD is important because it changes the optimization dynamics in a major way.

Compared with Batch GD, it usually:
- updates parameters more frequently
- introduces more noise into training
- can react faster to local information
- can behave less smoothly across updates

This makes SGD a fundamental optimization method to understand.

### Current implementation status

At the current stage of the project, the first deterministic SGD workflow is implemented.

The current project supports:
- sample-level gradient computation through `stochastic_gradient(...)`
- one SGD update through `stochastic_gradient_descent_step(...)`
- repeated SGD training through `fit_stochastic(...)`
- per-epoch history recording through `fit_stochastic_with_history(...)`
- explicit experiment packaging through `run_sgd_experiment(...)`

So SGD is now present in the project both as a training mechanism and as an optimization experiment.

### Relationship with Batch GD

Batch GD and SGD use the same core update rule idea:

$$
\theta \leftarrow \theta - \alpha \frac{\partial L}{\partial \theta}
$$

The difference is not the existence of the update rule.

The difference is the amount of data used to compute each update.

#### Batch GD
- uses the full dataset
- computes one average gradient
- performs one update per epoch

#### SGD
- uses one sample at a time
- computes one sample-level gradient
- performs many updates within one pass through the dataset

So SGD changes the granularity of optimization.

### Intuition

Batch GD moves using a smooth average direction computed from all data.

SGD moves using a much noisier local direction computed from one sample.

Because of this:
- Batch GD is usually easier to analyze
- SGD is usually less smooth
- SGD may move faster in some settings
- SGD may oscillate more even when training is progressing

This makes SGD especially important for understanding optimization as a dynamic process.

### Typical qualitative behavior

Compared with Batch GD, SGD often shows:
- noisier loss evolution
- noisier parameter trajectories
- faster short-term reaction
- less stable per-update behavior

So SGD is usually not studied by looking only at one update.

It is studied by observing the overall trend across many updates.

### Current empirical observations

The current implementation already shows the main qualitative SGD behaviors expected on the reference dataset.

#### Sample-level update behavior

For a single sample, SGD performs a valid local correction based only on that sample.

This confirms that SGD uses the same update-rule logic as Batch GD, but at a different data granularity.

#### Reasonable learning rate

With a learning rate such as `0.01`:
- MSE decreases clearly across epochs
- parameters move toward the target region
- training remains stable

#### Smaller learning rate

With a smaller learning rate such as `0.001`:
- MSE still decreases
- parameter movement is slower
- convergence is more gradual

#### Larger learning rate

With a larger learning rate such as `0.1`:
- training can improve very quickly on the current small dataset
- parameter movement becomes more aggressive
- optimization remains usable here, but is less conservative

These observations match the expected idea that SGD is usually noisier and more sensitive to update-scale choices than Batch GD.

### First scope in this project

At the current stage of the project, SGD should be introduced in the simplest possible form.

The first goal is not to implement advanced SGD variants.

The first goal is to understand:
- how SGD differs from Batch GD
- how one-sample updates change training behavior
- how learning rate interacts with noisier updates
- how optimization trajectories differ from batch training

### What SGD is not yet

For now, SGD does not include:
- momentum
- mini-batch training
- adaptive learning rates
- shuffling strategies beyond the simplest useful version
- advanced convergence theory

Those belong later.

At this stage, SGD should be studied first as the simplest one-sample version of Gradient Descent.

### Why SGD belongs in this sub-phase

SGD is the natural next concept after Batch GD because it provides the first strong comparison point in optimization.

By comparing Batch GD and SGD, it becomes easier to understand:
- smooth vs noisy optimization
- average-gradient updates vs sample-level updates
- stable convergence vs unstable local fluctuations
- how learning rate interacts with different update regimes

This makes SGD one of the most important concepts in the whole optimization sub-phase.

---

## SGD as an optimization experiment

At this stage of the project, SGD can now be run as an explicit optimization experiment.

One experiment contains:
- a chosen learning rate
- a chosen number of epochs
- the full per-epoch training history

This makes it possible to study SGD not only as code that updates parameters, but also as an optimization process that can be inspected and compared.

The current experiment-level view makes it possible to study:
- how MSE changes across epochs
- how `weight` changes across epochs
- how `bias` changes across epochs
- how learning rate changes SGD behavior
- how SGD compares with Batch GD on the same dataset

So SGD is now represented at two levels in the project:
- as a training mechanism
- as an optimization experiment

This is important because SGD becomes much more meaningful when it can be compared directly with Batch GD.

---

## Explicit learning rate experiments

### Purpose

Learning rate experiments are used to study how different learning-rate choices change optimization behavior.

The goal is not only to run training with a single value of \(\alpha\), but to compare multiple values in a structured way.

This makes it possible to observe how update scale affects:
- speed of improvement
- stability of training
- convergence quality
- divergence risk

### Why experiments are needed

The effect of the learning rate cannot be understood well from the update rule alone.

Even though the update formula is simple:

$$
\theta \leftarrow \theta - \alpha \frac{\partial L}{\partial \theta}
$$

small changes in \(\alpha\) can produce very different optimization trajectories.

So learning rate must be studied empirically, not only symbolically.

This is why explicit experiments are a natural and necessary part of the optimization sub-phase.

### Current implementation status

At the current stage of the project, explicit learning-rate studies are implemented for both Batch GD and SGD.

The current project supports:
- Batch GD learning-rate studies through `run_batch_gd_learning_rate_study(...)`
- SGD learning-rate studies through `run_sgd_learning_rate_study(...)`
- grouped experiment results across multiple learning rates
- compact comparison of final optimization outcomes across experiments

This means learning-rate analysis is no longer only a theoretical idea in the project. It is now also a structured experiment workflow.

### Core idea

A learning rate experiment means running the same optimization problem multiple times while changing only the learning rate.

So the goal is to isolate the effect of \(\alpha\).

For a fair comparison, the following should remain fixed:
- dataset
- initial parameters
- number of epochs
- optimization regime being tested

The only variable that changes should be the learning rate.

### What should be observed

In each experiment, the most important quantities to observe are:
- MSE across epochs
- parameter evolution across epochs
- final parameter values
- final MSE
- qualitative stability of the trajectory

The objective is not only to identify which run ends with the lowest loss, but also to understand how the optimization process behaves while getting there.

### Typical outcomes

#### Very small learning rate

A very small learning rate usually produces:
- slow parameter movement
- slow MSE decrease
- stable but inefficient optimization

#### Reasonable learning rate

A reasonable learning rate usually produces:
- clear loss reduction
- efficient progress
- stable trajectories
- useful convergence speed

#### Too large learning rate

A learning rate that is too large may produce:
- strong overshooting
- oscillation
- unstable parameter motion
- loss explosion or divergence

These outcomes are not only implementation details. They are central optimization behaviors.

### Current empirical observations

The current implementation already shows the expected qualitative patterns from explicit learning-rate comparison.

#### Batch GD study observations

With a representative study such as learning rates `0.01`, `0.1`, and `1.0`:
- `0.01` improves slowly
- `0.1` produces strong and stable progress
- `1.0` performs much worse and behaves unstably relative to the better choice

#### SGD study observations

With a representative study such as learning rates `0.001`, `0.01`, and `0.1`:
- `0.001` improves slowly
- `0.01` gives clear stable progress
- `0.1` can improve very quickly on the current small dataset, but behaves more aggressively

These observations match the theoretical idea that optimization behavior is highly sensitive to learning-rate choice.

### Why learning rate experiments matter for Batch GD and SGD

Learning rate experiments are important for both Batch GD and SGD, but the observations are not always the same.

#### For Batch GD

Batch GD often produces smoother trajectories, so the effect of the learning rate is easier to interpret visually and numerically.

#### For SGD

SGD is noisier by nature, so learning rate experiments help show how update-scale choices interact with sample-level noise.

This makes the same \(\alpha\) potentially feel different under Batch GD and SGD.

So learning rate experiments are also a bridge toward method comparison.

### First scope in this project

At the current stage, learning rate experiments should stay simple and focused.

The first goal is to compare a few representative values such as:
- small
- reasonable
- large

The first goal is not to perform automatic tuning or large hyperparameter sweeps.

The goal is to build intuition.

### What these experiments are not yet

For now, explicit learning rate experiments do not include:
- learning rate schedules
- decay strategies
- adaptive optimizers
- automatic search procedures
- hyperparameter optimization frameworks

Those belong later.

At this stage, the experiments should remain simple, controlled, and interpretable.

### Why this concept belongs now

Learning rate experiments naturally come after Batch GD and SGD are implemented.

Once both optimization regimes exist, it becomes meaningful to ask:
- how sensitive is each method to the learning rate?
- which method is smoother?
- which method improves faster under the same nominal \(\alpha\)?
- how do stability and convergence depend on update scale?

This makes learning rate experiments the right next concept before stability analysis and broader convergence comparison.

---

## Learning-rate studies as optimization experiments

At this stage of the project, learning-rate analysis can now be performed through explicit grouped studies.

One study contains:
- a fixed dataset
- a fixed epoch count
- a list of learning rates
- one experiment result per learning rate

This is important because it turns learning-rate comparison into a structured optimization object instead of a sequence of unrelated manual runs.

The current study-level view makes it possible to compare:
- final MSE across learning rates
- final parameter values across learning rates
- qualitative stability differences
- optimization behavior under Batch GD and SGD

So learning-rate analysis is now represented at two levels in the project:
- as a theoretical optimization concept
- as a structured experiment-comparison workflow

This is an important bridge toward the next concepts: stability analysis, convergence behavior, and Batch GD vs SGD comparison.

---

## Stability analysis

### Purpose

Stability analysis studies whether an optimization process behaves in a controlled and reliable way across repeated updates.

The main question is:

> Do the parameter updates remain well-behaved, or do they become erratic, oscillatory, or explosive?

In this sub-phase, stability analysis helps distinguish between:
- optimization that is progressing in a controlled way
- optimization that is numerically or dynamically problematic

So stability analysis focuses on the quality of the optimization trajectory, not only on whether the loss sometimes goes down.

### Why stability matters

A training process can reduce the loss at some moments and still be unstable overall.

For example:
- parameters may overshoot repeatedly
- loss may oscillate strongly
- updates may become too large
- optimization may diverge after a few steps

So it is not enough to ask only:
- is the loss lower than before?

It is also necessary to ask:
- is the optimization process behaving in a controlled way?

This is why stability is one of the central concepts in optimization.

### Current implementation status

At the current stage of the project, the first empirical stability-analysis workflow is implemented.

The current project supports:
- stability classification through `StabilityStatus`
- compact per-experiment summaries through `StabilityAnalysisResult`
- Batch GD stability analysis through `analyze_batch_gd_stability(...)`
- SGD stability analysis through `analyze_sgd_stability(...)`

This means stability analysis is no longer only a theoretical idea in the project. It is now also an explicit interpretation layer built on top of optimization experiment history.

### Stability vs convergence

Stability and convergence are closely related, but they are not the same idea.

#### Stability

Stability asks whether the optimization trajectory behaves in a controlled way.

It is about the quality of the update process.

#### Convergence

Convergence asks whether the optimization process moves toward a better or more stable solution over time.

It is about the long-term destination of the trajectory.

So stability is more local and behavioral, while convergence is more global and long-term.

A process can be:
- stable and convergent
- stable but very slow
- unstable and non-convergent
- temporarily improving but not truly stable

This is why the two concepts should be studied separately.

### What stability looks like in practice

For the current project, stable optimization usually means:
- MSE does not explode
- parameter values do not grow uncontrollably
- updates do not oscillate violently
- training remains interpretable across epochs

This does not mean every update must be tiny or monotonic.

It means the overall behavior remains controlled.

### Typical signs of stable behavior

A stable optimization process often shows:
- loss values that remain bounded
- parameters that move within a reasonable range
- progressively smaller corrections near a good solution
- trajectories that remain interpretable across epochs

For Batch GD, stable behavior is often smoother.

For SGD, stable behavior may still look noisy, but the noise should remain bounded rather than explosive.

### Typical signs of instability

Instability often appears as one or more of the following:
- exploding loss
- exploding parameter magnitude
- repeated overshooting
- strong oscillations without settling
- sensitivity to small hyperparameter changes

These are important warning signs because they show that the optimizer is not behaving robustly.

### Current empirical observations

The current implementation already shows the main qualitative stability cases expected from the project experiments.

#### Stable Batch GD case

For a reasonable Batch GD run:
- the loss remains bounded
- the final MSE is lower than the initial recorded MSE
- parameter magnitudes remain bounded
- the run is classified as `Stable`

#### Divergent Batch GD case

For an overly aggressive Batch GD run:
- MSE grows explosively
- parameter magnitudes grow explosively
- the run is classified as `Divergent`

#### Stable SGD case

For a reasonable SGD run:
- the loss decreases strongly across epochs
- parameter values remain bounded
- the run is classified as `Stable`

#### Divergent SGD case

For an overly aggressive SGD run:
- MSE explodes
- parameter values grow to extremely large magnitudes
- the run is classified as `Divergent`

These observations match the theoretical distinction between controlled and uncontrolled optimization behavior.

### Relationship with learning rate

The learning rate is one of the most important factors in stability.

If the learning rate is too large:
- updates may overshoot
- oscillations may appear
- divergence may occur

If the learning rate is smaller:
- optimization is often more stable
- but progress may become slower

So stability analysis is naturally connected to learning-rate experiments.

### Stability in Batch GD vs SGD

Stability does not look identical under Batch GD and SGD.

#### Batch GD

Batch GD usually produces smoother trajectories because each update is based on the full dataset.

So stability problems are often easier to see directly in:
- exploding loss
- large oscillations
- strong overshooting

#### SGD

SGD is noisier by design because each update is based on one sample.

So stability analysis in SGD requires more care.

A noisy trajectory is not automatically unstable.

The important question is whether the noise remains bounded and whether the overall process still behaves in a controlled way.

This makes stability analysis especially useful when comparing Batch GD and SGD.

### First scope in this project

At the current stage, stability analysis should remain simple and empirical.

The first goal is to observe:
- whether loss remains bounded
- whether parameters remain bounded
- whether trajectories remain interpretable
- whether different learning rates produce stable or unstable behavior

The first goal is not to develop a full mathematical stability theory.

The goal is to build practical intuition.

### What stability analysis is not yet

For now, stability analysis does not include:
- formal Lyapunov analysis
- spectral analysis of update operators
- advanced numerical optimization theory
- second-order methods
- rigorous stability proofs

Those belong much later.

At this stage, stability analysis should remain focused on controlled empirical observation.

### Why this concept belongs now

Stability analysis naturally comes after:
- Batch GD experiments
- SGD experiments
- explicit learning-rate studies

Once these mechanisms exist, it becomes possible to ask not only whether optimization improves the model, but also whether it does so robustly.

That is why stability analysis is the right next concept before deeper convergence comparison.

---

## Stability analysis as an experiment-interpretation layer

At this stage of the project, stability analysis is performed by interpreting already-recorded optimization histories.

This is important because stability does not require a new optimizer. It requires a structured way to read the behavior of an optimizer.

The current stability-analysis layer makes it possible to summarize an experiment in terms of:
- whether the run stayed bounded
- whether the final loss improved or deteriorated
- how large parameter magnitudes became during training
- whether the run should be considered stable, unstable, or divergent

So stability analysis is now represented at two levels in the project:
- as a theoretical optimization concept
- as a concrete interpretation layer applied to Batch GD and SGD experiment results

This is an important bridge toward the next concept, because convergence behavior depends not only on whether the loss eventually improves, but also on whether the optimization process remains sufficiently well-behaved along the way.

---

## Convergence behavior

### Purpose

Convergence behavior studies how an optimization process evolves over time toward a better solution.

The main question is:

> Does the optimizer move progressively toward a useful parameter region, and how does that progress look across training?

In this sub-phase, convergence behavior helps analyze not only whether optimization is stable, but also how efficiently and in what manner it improves the model.

So convergence behavior focuses on the long-term trend of the optimization trajectory.

### Why convergence behavior matters

A training process may be stable and still converge very slowly.

A training process may improve quickly at first and then plateau.

A training process may oscillate while still trending toward a good region.

So stability alone is not enough to understand optimization quality.

It is also necessary to study:
- how fast improvement happens
- whether improvement is sustained
- whether the optimizer approaches a useful solution region
- whether progress slows down, stalls, or reverses

This is why convergence behavior is one of the central concepts in optimization.

### Current implementation status

At the current stage of the project, the first empirical convergence-analysis workflow is implemented.

The current project supports:
- convergence classification through `ConvergenceStatus`
- compact per-experiment summaries through `ConvergenceAnalysisResult`
- Batch GD convergence analysis through `analyze_batch_gd_convergence(...)`
- SGD convergence analysis through `analyze_sgd_convergence(...)`

This means convergence behavior is no longer only a theoretical idea in the project. It is now also an explicit interpretation layer built on top of optimization experiment history.

### Convergence vs stability

Convergence and stability are closely related, but they answer different questions.

#### Stability

Stability asks whether the trajectory remains well-behaved.

It is about control and boundedness.

#### Convergence

Convergence asks whether the trajectory moves toward a better solution over time.

It is about long-term progress.

So stability is about whether optimization behaves safely, while convergence is about whether optimization behaves productively.

A process can be:
- stable and convergent
- stable but slow
- stable but stalled
- unstable and non-convergent
- noisy but still convergent in trend

This is why convergence behavior deserves its own analysis.

### What convergence looks like in practice

For the current project, good convergence behavior usually means:
- MSE decreases across epochs in an overall downward trend
- parameter values move toward a useful solution region
- updates become less dramatic as training progresses
- optimization remains interpretable over time

This does not mean every epoch must improve perfectly.

Especially for SGD, local fluctuations may still be present.

What matters is the overall trend.

### Typical signs of good convergence

A run with good convergence behavior often shows:
- strong early improvement
- continued reduction in loss over time
- parameter trajectories that approach a stable region
- smaller corrections near a good solution

For simple problems, this may look like rapid MSE decrease followed by finer refinement.

### Typical signs of poor convergence

Poor convergence behavior may appear as:
- very slow improvement
- long plateaus
- progress that stalls far from a good solution
- oscillations that prevent settling
- deterioration after initial improvement

These signs are important because a run can be stable but still converge poorly.

### Current empirical observations

The current implementation already shows the main qualitative convergence cases expected from the project experiments.

#### Strong Batch GD case

For a reasonable Batch GD run:
- the final MSE is much lower than the initial recorded MSE
- relative improvement is very large
- the run is classified as `Strong`

#### Failed Batch GD case

For an overly aggressive Batch GD run:
- the final MSE is much worse than the initial recorded MSE
- relative improvement becomes negative
- the run is classified as `Failed`

#### Strong SGD case

For a strong SGD run:
- the final MSE is much lower than the initial recorded MSE
- relative improvement is very large
- the run is classified as `Strong`

#### Slow SGD case

For a weaker but still improving SGD run:
- the final MSE is lower than the initial recorded MSE
- progress is meaningful but not strong enough to enter the top category
- the run is classified as `Slow`

These observations match the theoretical distinction between stronger and weaker forms of convergence progress.

### Relationship with learning rate

The learning rate has a major influence on convergence behavior.

If the learning rate is too small:
- progress may be very slow
- convergence may take many epochs

If the learning rate is reasonable:
- progress may be both stable and efficient
- the optimizer may approach a good region quickly

If the learning rate is too large:
- convergence may fail
- overshooting may prevent progress
- divergence may occur

So convergence behavior is strongly tied to learning-rate choice.

### Convergence in Batch GD vs SGD

Convergence does not look identical under Batch GD and SGD.

#### Batch GD

Batch GD often shows smoother convergence because each update is based on the full dataset.

This makes the loss trajectory easier to interpret across epochs.

#### SGD

SGD often shows noisier convergence because each update is based on one sample.

So the per-epoch trend may still improve overall even when local movement is less smooth.

This means convergence analysis for SGD should focus more on trend than on perfectly monotonic behavior.

### First scope in this project

At the current stage, convergence behavior should be studied empirically and simply.

The first goal is to observe:
- how MSE changes across epochs
- how parameters move toward solution regions
- how Batch GD and SGD differ in convergence style
- how learning rate affects convergence speed and quality

The first goal is not to develop full theoretical convergence proofs.

The goal is to build intuition from controlled experiments.

### What convergence analysis is not yet

For now, convergence behavior does not include:
- formal proof of convergence conditions
- convex-optimization convergence theorems
- asymptotic rate proofs
- second-order convergence analysis
- theoretical guarantees beyond the current empirical scope

Those belong much later.

At this stage, convergence behavior should remain focused on observed training dynamics.

### Why this concept belongs now

Convergence behavior naturally comes after:
- Batch GD experiments
- SGD experiments
- explicit learning-rate studies
- stability analysis

Once those pieces exist, it becomes possible to ask not only whether optimization stays controlled, but also whether it actually moves toward a good solution efficiently.

That is why convergence behavior is the right next concept before a broader Batch GD vs SGD comparison.

---

## Convergence analysis as an experiment-interpretation layer

At this stage of the project, convergence analysis is performed by interpreting already-recorded optimization histories.

This is important because convergence does not require a new optimizer. It requires a structured way to summarize long-term progress.

The current convergence-analysis layer makes it possible to summarize an experiment in terms of:
- whether the run made strong, slow, stalled, or failed progress
- how much relative loss reduction happened
- where the optimizer ended in parameter space
- how much progress was achieved from the first recorded epoch to the last

So convergence behavior is now represented at two levels in the project:
- as a theoretical optimization concept
- as a concrete interpretation layer applied to Batch GD and SGD experiment results

This is an important bridge toward the final comparison concept, because Batch GD vs SGD comparison depends not only on stability, but also on the style and quality of convergence.

---

## Why this sub-phase matters

This sub-phase is important because it develops intuition that will be reused throughout the rest of the project.

Understanding optimization means understanding:
- why training works
- why training sometimes fails
- how learning rate changes behavior
- how update rules shape convergence

This is one of the central foundations of Machine Learning, not only of Deep Learning.
