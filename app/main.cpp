#include "ml/linear_regression/linear_regression.hpp"
#include "ml/logistic_regression/logistic_regression.hpp"
#include "ml/common/vector_ops.hpp"
#include "ml/common/csv_loader.hpp"
#include "ml/common/data_split.hpp"
#include "ml/common/feature_scaling.hpp"
#include "ml/common/activation_functions.hpp"
#include "ml/common/loss_functions.hpp"

#include <iostream>
#include <vector>

template <typename T>
void print_vector(const std::vector<T>& v) {
    std::cout <<"[ ";
    for (std::size_t i = 0; i < v.size(); ++i) {
        std::cout << v[i];
        if (i + 1 < v.size()) {
            std::cout << ", ";
        }
    }
    std::cout << " ]\n";
}

void print_dataset(const ml::common::Dataset& dataset, const std::string& name) {
    std::cout << name << " features:\n";
    for (const auto& row : dataset.features) {
        print_vector(row);
    }
    
    std::cout << name << " targets:\n";
    print_vector(dataset.targets);
}

void print_snapshot(const ml::TrainingSnapshot& snapshot) {
    std::cout << "Epoch:  " << snapshot.epoch << "\n";
    std::cout << "MSE:    " << snapshot.mse << "\n";
    std::cout << "Weight: " << snapshot.weight << "\n";
    std::cout << "Bias:   " << snapshot.bias<< "\n";
}

void print_history(const std::vector<ml::TrainingSnapshot>& history) {
    for (const auto& snapshot : history) {
        print_snapshot(snapshot);
        std::cout << "--------------------\n";
    }
}

void print_model(const ml::LinearRegression& model) {
    std::cout << "Model parameters:\n";
    std::cout << "weight = " << model.weight() << "\n";
    std::cout << "bias   = " << model.bias() << "\n";
}

void print_batch_gd_experiment_summary(const ml::BatchGDExperimentResult& result) {
    std::cout << "Learning rate: " << result.learning_rate << "\n";
    std::cout << "Epochs: " << result.epochs << "\n";

    if (result.history.empty()) {
        std::cout << "History: empty\n";
        return;
    }

    const ml::TrainingSnapshot& last = result.history.back();
    std::cout << "Final epoch:  " << last.epoch << "\n";
    std::cout << "Final MSE:    " << last.mse << "\n";
    std::cout << "Final weight: " << last.weight << "\n";
    std::cout << "Final bias:   " << last.bias << "\n";
}

void print_sgd_experiment_summary(const ml::SGDExperimentResult& result) {
    std::cout << "Learning rate: " << result.learning_rate << "\n";
    std::cout << "Epochs: " << result.epochs << "\n";

    if (result.history.empty()) {
        std::cout << "History: empty\n";
        return;
    }

    const ml::TrainingSnapshot& last = result.history.back();
    std::cout << "Final epoch:  " << last.epoch << "\n";
    std::cout << "Final MSE:    " << last.mse << "\n";
    std::cout << "Final weight: " << last.weight << "\n";
    std::cout << "Final bias:   " << last.bias << "\n";
}

void print_batch_gd_learning_rate_study(const ml::BatchGDLearningRateStudyResult& study) {
    std::cout << "Batch GD learning-rate study\n";
    std::cout << "Epochs: " << study.epochs << "\n";
    std::cout << "Number of experiments: " << study.experiments.size() << "\n\n";

    for (const auto& experiment : study.experiments) {
        print_batch_gd_experiment_summary(experiment);
        std::cout << "--------------------\n";
    }
}

void print_sgd_learning_rate_study(const ml::SGDLearningRateStudyResult& study) {
    std::cout << "SGD learning-rate study\n";
    std::cout << "Epochs: " << study.epochs << "\n";
    std::cout << "Number of experiments: " << study.experiments.size() << "\n\n";

    for (const auto& experiment : study.experiments) {
        print_sgd_experiment_summary(experiment);
        std::cout << "--------------------\n";
    }
}

const char* stability_status_to_string(ml::StabilityStatus status) {
    switch (status) {
        case ml::StabilityStatus::Stable:
            return "Stable";
        case ml::StabilityStatus::Unstable:
            return "Unstable";
        case ml::StabilityStatus::Divergent:
            return "Divergent";
        default:
            return "Unknown";
    }
}

void print_stability_analysis(const ml::StabilityAnalysisResult& result) {
    std::cout << "Stability status: " << stability_status_to_string(result.status) << "\n";
    std::cout << "Initial MSE:      " << result.initial_mse << "\n";
    std::cout << "Final MSE:        " << result.final_mse << "\n";
    std::cout << "Max |weight|:     " << result.max_abs_weight << "\n";
    std::cout << "Max |bias|:       " << result.max_abs_bias << "\n";
}

const char* convergence_status_to_string(ml::ConvergenceStatus status) {
    switch (status) {
        case ml::ConvergenceStatus::Strong:
            return "Strong";
        case ml::ConvergenceStatus::Slow:
            return "Slow";
        case ml::ConvergenceStatus::Stalled:
            return "Stalled";
        case ml::ConvergenceStatus::Failed:
            return "Failed";
        default:
            return "Unknown";
    }
}

void print_convergence_analysis(const ml::ConvergenceAnalysisResult& result) {
    std::cout << "Convergence status:   " << convergence_status_to_string(result.status) << "\n";
    std::cout << "Initial MSE:          " << result.initial_mse << "\n";
    std::cout << "Final MSE:            " << result.final_mse << "\n";
    std::cout << "Relative improvement: " << result.relative_improvement << "\n";
    std::cout << "Final weight:         " << result.final_weight << "\n";
    std::cout << "Final bias:           " << result.final_bias << "\n";
}

void print_feature_variant(const std::string& name, const std::vector<double>& values) {
    std::cout << name << ": ";
    print_vector(values);
}

void print_batch_gd_compact_comparison(
    ml::LinearRegression& model,
    const std::string& name,
    const std::vector<double>& x,
    const std::vector<double>& y,
    double learning_rate,
    std::size_t epochs) {
    model.set_parameters(0.0, 0.0);

    ml::BatchGDExperimentResult experiment =
        model.run_batch_gd_experiment(x, y, learning_rate, epochs);
    ml::StabilityAnalysisResult stability =
        model.analyze_batch_gd_stability(experiment);
    ml::ConvergenceAnalysisResult convergence =
        model.analyze_batch_gd_convergence(experiment);

    const ml::TrainingSnapshot& last = experiment.history.back();

    std::cout << "\nBatch GD comparison – " << name << ":\n";
    std::cout << "Learning rate:        " << learning_rate << "\n";
    std::cout << "Epochs:               " << epochs << "\n";
    std::cout << "Final MSE:            " << last.mse << "\n";
    std::cout << "Final weight:         " << last.weight << "\n";
    std::cout << "Final bias:           " << last.bias << "\n";
    std::cout << "Stability status:     " << stability_status_to_string(stability.status) << "\n";
    std::cout << "Convergence status:   " << convergence_status_to_string(convergence.status) << "\n";
    std::cout << "Relative improvement: " << convergence.relative_improvement << "\n";
}

int main() {
    // ---------------------
    // LINEAR REGRESSION TEST
    // ---------------------

    // ml::LinearRegression model;

    // model.set_parameters(2.0, 1.0);

    // double x = 3.0;
    // double y = model.predict(x);

    // std::cout << "Single prediction:\n";
    // std::cout << "x = " << x << ", y = " << y << "\n\n";

    // std::vector<double> xs = {0.0, 1.0, 2.0, 3.0, 4.0};
    // std::vector<double> ys = model.predict(xs);
    
    // std::cout << "Batch prediction:\n";
    // for (std::size_t i = 0; i < xs.size() ; ++i) {
    //     std::cout << "x = " << xs[i] << ", y = " << ys[i] << "\n";
    // }
    
    // std::cout << "\nModel parameters:\n";
    // std::cout << "weight = " << model.weight() << "\n";
    // std::cout << "bias   = " << model.bias() << "\n";


    // ---------------------
    // VECTOR OPS TEST
    // ---------------------
    
    // std::vector<double> a{1.0, 2.0, 3.0};
    // std::vector<double> b{4.0, 5.0, 6.0};
    
    // std::vector<double> sum = ml::common::add(a, b);
    // std::vector<double> subtract = ml::common::subtract(a, b);
    // std::vector<double> scaled = ml::common::scalar_multiply(a, 2.0);
    // double dot = ml::common::dot(a, b);
    // double mean = ml::common::mean(a);
    
    // std::cout << "sum: ";
    // print_vector(sum);
    
    // std::cout << "subtract: ";
    // print_vector(subtract);
    
    // std::cout << "scaled: ";
    // print_vector(scaled);
    
    // std::cout << "dot: ";
    // std::cout << dot << "\n";
    
    // std::cout << "mean: ";
    // std::cout << mean << "\n";
    
    // try {
    //     std::vector<double> c{1.0, 2.0};
    //     auto bad = ml::common::add(a, c);
    //     print_vector(bad);
    // } catch (const std::invalid_argument& e) {
    //     std::cout << "Caught error: " << e.what() << "\n";
    // }
    
    // try {
    //     std::vector<double> d{};
    //     auto bad = ml::common::mean(d);
    //     std::cout << bad << "\n";
    // } catch (const std::invalid_argument& e) {
    //     std::cout << "Caught error: " << e.what() << "\n";
    // }
    

    // ---------------------
    // CSV LOADER TEST
    // ---------------------
    
    // try {
    //     ml::common::Dataset dataset = ml::common::load_csv("data/test_linear.csv");
    //     std::cout << "\nLoaded dataset successfully:\n";
    //     print_dataset(dataset, "Test Database");
    // } catch (const std::runtime_error& e) {
    //     std::cout << "CSV loading error: " << e.what() << "\n";
    // }
    
    
    // ---------------------
    // SPLIT DATASET TEST
    // ---------------------

    // try {
    //     ml::common::Dataset dataset = ml::common::load_csv("data/test_linear.csv");

    //     auto [train_dataset, test_dataset] = ml::common::train_test_split(dataset, 0.75);

    //     std::cout << "Full dataset loaded successfully.\n\n";

    //     print_dataset(train_dataset, "Train");
    //     std::cout << "\n";
    //     print_dataset(test_dataset, "Test");
    // } catch (const std::exception& e) {
    //     std::cout << "Error: " << e.what() << "\n";
    // }


    // ---------------------
    // LINEAR REGRESSION::MSE TEST
    // ---------------------
    
    // ml::LinearRegression model;
    
    // // model.set_parameters(2.0, 1.0);
    // model.set_parameters(0.0, 0.0);
    
    // std::cout << "Model parameters:\n";
    // std::cout << "weight = " << model.weight() << "\n";
    // std::cout << "bias   = " << model.bias() << "\n";
    
    // std::vector<double> xs = {1.0, 2.0, 3.0};
    // std::vector<double> ys = {3.0, 5.0, 7.0};
    
    // std::cout << "\nMSE: " << model.mse(xs, ys) << "\n";
    
    // std::cout << "\nGradients:\n";
    // std::cout << model.gradient(xs, ys).first << "\n";
    // std::cout << model.gradient(xs, ys).second << "\n";
    
    // double learning_rate = 0.1;
    
    // std::cout << "\nGDS:\n";
    // model.gradient_descent_step(xs, ys, learning_rate);
    
    // std::cout << "Model parameters:\n";
    // std::cout << "weight = " << model.weight() << "\n";
    // std::cout << "bias   = " << model.bias() << "\n";
    
    // std::cout << "\nMSE: " << model.mse(xs, ys) << "\n";
    
    // std::size_t epochs = 10;
    
    // std::cout << "\nFit Batch:\n";
    // model.fit_batch(xs, ys, learning_rate, epochs);
    
    // std::cout << "Model parameters:\n";
    // std::cout << "weight = " << model.weight() << "\n";
    // std::cout << "bias   = " << model.bias() << "\n";
    
    // std::cout << "\nMSE: " << model.mse(xs, ys) << "\n";
    
    
    // ---------------------
    // LINEAR REGRESSION Convergence analysis for Linear Regression with batch GD TEST
    // ---------------------
    // ml::LinearRegression model;
    
    // // model.set_parameters(2.0, 1.0);
    // model.set_parameters(0.0, 0.0);
    
    // std::cout << "Model parameters:\n";
    // std::cout << "weight = " << model.weight() << "\n";
    // std::cout << "bias   = " << model.bias() << "\n";
    
    // std::vector<double> xs = {1.0, 2.0, 3.0};
    // std::vector<double> ys = {3.0, 5.0, 7.0};
    
    // std::cout << "\nMSE: " << model.mse(xs, ys) << "\n";

    // double learning_rate = 0.1;
    // std::size_t epochs = 5;

    // std::cout << "\nFit Batch:\n";
    // model.fit_batch(xs, ys, learning_rate, epochs);
    
    // std::cout << "Model parameters:\n";
    // std::cout << "weight = " << model.weight() << "\n";
    // std::cout << "bias   = " << model.bias() << "\n";
    
    // std::cout << "\nMSE: " << model.mse(xs, ys) << "\n";
    
    // std::cout << "\nConvergence analysis with batch GD:\n";
    // std::vector<ml::TrainingSnapshot> history = model.fit_batch_with_history(xs, ys, learning_rate, epochs);
    // print_history(history);
    
    // std::cout << "Epochs: " << epochs << "\n";
    // std::cout << "History size: " << history.size() << "\n";

    // std::cout << "\nBatch GD experiment:\n";
    // ml::BatchGDExperimentResult batchGDExperimentResult = model.run_batch_gd_experiment(xs, ys, learning_rate, epochs);
    // std::cout << "Learning rate: " << batchGDExperimentResult.learning_rate << "\n";
    // std::cout << "Epochs: " << batchGDExperimentResult.epochs << "\n";
    // print_history(batchGDExperimentResult.history);


    // // ---------------------
    // // LINEAR REGRESSION – SGD Experiment
    // // ---------------------
    // ml::LinearRegression model;
    
    // // model.set_parameters(2.0, 1.0);
    // model.set_parameters(0.0, 0.0);
    
    // std::cout << "Model parameters:\n";
    // std::cout << "weight = " << model.weight() << "\n";
    // std::cout << "bias   = " << model.bias() << "\n";
    
    // std::vector<double> xs = {1.0, 2.0, 3.0};
    // std::vector<double> ys = {3.0, 5.0, 7.0};
    
    // std::cout << "\nMSE: " << model.mse(xs, ys) << "\n";

    // // double sample_x = 1.0;
    // // double sample_y = 3.0;
    
    // // std::cout << "\nSGD gradients:\n";
    // // std::cout << model.stochastic_gradient(sample_x, sample_y).first << "\n";
    // // std::cout << model.stochastic_gradient(sample_x, sample_y).second << "\n";

    // double learning_rate = 0.1;
    // std::size_t epochs = 10;
    
    // // std::cout << "\nSGD step:\n";
    // // model.stochastic_gradient_descent_step(sample_x, sample_y, learning_rate);
    
    // // std::cout << "\nFit SGD:\n";
    // // model.fit_stochastic(xs, ys, learning_rate, epochs);
    
    // // std::cout << "Model parameters:\n";
    // // std::cout << "weight = " << model.weight() << "\n";
    // // std::cout << "bias   = " << model.bias() << "\n";
    
    // std::cout << "\nFit SGD with History:\n";
    // std::vector<ml::TrainingSnapshot> history = model.fit_stochastic_with_history(xs, ys, learning_rate, epochs);
    // std::cout << "History: " << history.size() << "\n";
    // print_history(history);

    // // std::cout << "\nBatch GD experiment:\n";
    // // ml::BatchGDExperimentResult batchGDExperimentResult = model.run_batch_gd_experiment(xs, ys, learning_rate, epochs);
    // // // std::cout << "Learning rate: " << batchGDExperimentResult.learning_rate << "\n";
    // // // std::cout << "Epochs: " << batchGDExperimentResult.epochs << "\n";
    // // print_history(batchGDExperimentResult.history);


    // // ---------------------
    // // LINEAR REGRESSION – Learning rate experiments
    // // ---------------------
    // ml::LinearRegression model;
    
    // // model.set_parameters(2.0, 1.0);
    // model.set_parameters(0.0, 0.0);
    
    // std::cout << "Model parameters:\n";
    // std::cout << "weight = " << model.weight() << "\n";
    // std::cout << "bias   = " << model.bias() << "\n";
    
    // std::vector<double> xs = {1.0, 2.0, 3.0};
    // std::vector<double> ys = {3.0, 5.0, 7.0};
    
    // std::cout << "\nMSE: " << model.mse(xs, ys) << "\n";

    // std::vector<double> learning_rates = {0.01, 0.1, 1.0};
    // std::size_t epochs = 5;

    // // std::cout << "\nBatch GD Learning Rate Study:\n";
    // // ml::BatchGDLearningRateStudyResult batchGDstudy = model.run_batch_gd_learning_rate_study(xs, ys, learning_rates, epochs);
    // // print_batch_gd_learning_rate_study(batchGDstudy);
    // // std::cout << "learning_rates size: " << learning_rates.size() << "\n";
    // // std::cout << "batchGDstudy.experiments size: " << batchGDstudy.experiments.size() << "\n";
    
    // std::cout << "\nSGD Learning Rate Study:\n";
    // ml::SGDLearningRateStudyResult SGDstudy = model.run_sgd_learning_rate_study(xs, ys, learning_rates, epochs);
    // print_sgd_learning_rate_study(SGDstudy);
    // std::cout << "learning_rates size: " << learning_rates.size() << "\n";
    // std::cout << "SGDstudy.experiments size: " << SGDstudy.experiments.size() << "\n";


    // // ---------------------
    // // LINEAR REGRESSION – Stability analysis
    // // ---------------------
    // ml::LinearRegression model;
    
    // // model.set_parameters(2.0, 1.0);
    // model.set_parameters(0.0, 0.0);
    
    // std::cout << "Model parameters:\n";
    // std::cout << "weight = " << model.weight() << "\n";
    // std::cout << "bias   = " << model.bias() << "\n";
    
    // std::vector<double> xs = {1.0, 2.0, 3.0};
    // std::vector<double> ys = {3.0, 5.0, 7.0};
    
    // std::cout << "\nMSE: " << model.mse(xs, ys) << "\n";

    // double learning_rate = 0.001;
    // std::size_t epochs = 5;

    // std::cout << "\nBatch GD Stability Analysis:\n";
    // ml::BatchGDExperimentResult batchGDExperimentResult = model.run_batch_gd_experiment(xs, ys, learning_rate, epochs);
    // ml::StabilityAnalysisResult batchGDResult = model.analyze_batch_gd_stability(batchGDExperimentResult);
    // print_stability_analysis(batchGDResult);
    
    // // std::cout << "\nSGD Stability Analysis:\n";
    // // ml::SGDExperimentResult SGDExperimentResult = model.run_sgd_experiment(xs, ys, learning_rate, epochs);
    // // ml::StabilityAnalysisResult SGDResult = model.analyze_sgd_stability(SGDExperimentResult);
    // // print_stability_analysis(SGDResult);


    // // ---------------------
    // // LINEAR REGRESSION – Convergence analysis
    // // ---------------------
    // ml::LinearRegression model;
    
    // // model.set_parameters(2.0, 1.0);
    // model.set_parameters(0.0, 0.0);
    
    // std::cout << "Model parameters:\n";
    // std::cout << "weight = " << model.weight() << "\n";
    // std::cout << "bias   = " << model.bias() << "\n";
    
    // std::vector<double> xs = {1.0, 2.0, 3.0};
    // std::vector<double> ys = {3.0, 5.0, 7.0};
    
    // std::cout << "\nMSE: " << model.mse(xs, ys) << "\n";

    // double learning_rate = 0.01;
    // std::size_t epochs = 20;

    // std::cout << "\nBatch GD Convergence Analysis:\n";
    // ml::BatchGDExperimentResult batchGDExperimentResult = model.run_batch_gd_experiment(xs, ys, learning_rate, epochs);
    // ml::ConvergenceAnalysisResult batchGDResult = model.analyze_batch_gd_convergence(batchGDExperimentResult);
    // print_convergence_analysis(batchGDResult);
    
    // // std::cout << "\nSGD Convergence Analysis:\n";
    // // ml::SGDExperimentResult SGDExperimentResult = model.run_sgd_experiment(xs, ys, learning_rate, epochs);
    // // ml::ConvergenceAnalysisResult SGDResult = model.analyze_sgd_convergence(SGDExperimentResult);
    // // print_convergence_analysis(SGDResult);


    // // ---------------------
    // // FEATURE SCALING – Min Max Scaling
    // // ---------------------
    // // std::vector<double> values = {10, 15, 20};
    // std::vector<double> values = {10, 12, 15, 17, 20};
    // // std::vector<double> values = {-10, 0, 10};
    // // std::vector<double> values = {2, 4, 8};
    // // std::vector<double> values = {2, 2, 2};
    // // std::vector<double> values = {};

    // std::cout << "Min Max Scaling:" << "\n";
    // try {
    //     std::vector<double> scaled_values = ml::common::min_max_scale(values);
    //     std::cout << "Scaled values: ";
    //     print_vector(scaled_values);
    // } catch (const std::invalid_argument& e) {
    //     std::cout << "Caught error: " << e.what() << "\n";
    // }
    
    
    // // ---------------------
    // // FEATURE SCALING – Standardize
    // // ---------------------
    // // std::vector<double> values = {10, 15, 20};
    // // std::vector<double> values = {-1, 0, 1};
    // std::vector<double> values = {2, 4, 6};
    // // std::vector<double> values = {1, 2, 10};
    // // std::vector<double> values = {10, 12, 15, 17, 20};
    // // std::vector<double> values = {};
    // // std::vector<double> values = {5, 5, 5};

    // std::cout << "Standardization:" << "\n";
    // try {
    //     std::vector<double> standardized = ml::common::standardize(values);
    //     std::cout << "Standardized values: ";
    //     print_vector(standardized);
    //     std::cout << "Standardized size: " << standardized.size() << "\n";
    //     std::cout << "Values size: " << values.size() << "\n";
    // } catch (const std::invalid_argument& e) {
    //     std::cout << "Caught error: " << e.what() << "\n";
    // }


    // // ---------------------
    // // FEATURE SCALING – Compare results
    // // ---------------------
    // ml::LinearRegression model;
    
    // std::vector<double> x = {10, 20, 30, 40};
    // std::vector<double> y = {21, 41, 61, 81};

    // double learning_rate = 0.1;
    // std::size_t epochs = 20;

    // std::cout << "Compare results:\n";
    // try {
    //     std::vector<double> x_raw = x;
    //     std::vector<double> x_minmax = ml::common::min_max_scale(x_raw);
    //     std::vector<double> x_standardized = ml::common::standardize(x_raw);

    //     std::cout << "\nFeature variants:\n";
    //     print_feature_variant("x_raw", x_raw);
    //     print_feature_variant("x_minmax", x_minmax);
    //     print_feature_variant("x_standardized", x_standardized);

    //     print_batch_gd_compact_comparison(model, "x_raw", x_raw, y, learning_rate, epochs);
    //     print_batch_gd_compact_comparison(model, "x_minmax", x_minmax, y, learning_rate, epochs);
    //     print_batch_gd_compact_comparison(model, "x_standardized", x_standardized, y, learning_rate, epochs);
    // } catch (const std::invalid_argument& e) {
    //     std::cout << "Caught error: " << e.what() << "\n";
    // }


    // // ---------------------
    // // ACTIVATION FUNCTIONS – Sigmoid
    // // ---------------------
    // std::vector<double> z_values = {0, 2, -2};
    
    // for (double z : z_values) {
    //     double probability = ml::common::sigmoid(z);
    
    //     std::cout << "\nSigmoid function:\n";
    //     std::cout << "z: " << z << "\n";
    //     std::cout << "Probability: " << probability << "\n";
    //     std::cout << "Binary predict: " << ml::common::binary_predict(probability) << "\n";
    // }
    
    
    // // ---------------------
    // // LOSS FUNCTIONS – Cross-entropy loss
    // // ---------------------
    // // std::vector<double> y_true = {1, 1, 0, 0};
    // // std::vector<double> y_pred = {0.9, 0.1, 0.1, 0.9};
    
    // // for (std::size_t i = 0; i < y_true.size(); ++i) {
    // //     std::cout << "\nCross-entropy loss function:\n";
    // //     std::cout << "y_true: " << y_true[i] << "\n";
    // //     std::cout << "y_pred: " << y_pred[i] << "\n";
    // //     std::cout << "Cross-entropy Loss: " << ml::common::binary_cross_entropy(y_true[i], y_pred[i]) << "\n";
    // // }
    
    // // std::vector<double> ys_true = {1, 0};
    // // std::vector<double> ys_pred = {0.9, 0.1};

    // // std::cout << "\nCross-entropy loss function:\n";
    // // std::cout << "y_true: ";
    // // print_vector(ys_true);
    // // std::cout << "y_pred: ";
    // // print_vector(ys_pred);
    // // std::cout << "Cross-entropy Loss: " << ml::common::binary_cross_entropy(ys_true, ys_pred) << "\n";
    
    // // double y_true = 2.0;
    // // double y_pred = 0.1;
    // double y_true = 0;
    // double y_pred = 0;
    
    // std::cout << "\nCross-entropy loss function:\n";
    // std::cout << "y_true: " << y_true << "\n";
    // std::cout << "y_pred: " << y_pred << "\n";

    // try {
    //     std::cout << "Cross-entropy Loss: " << ml::common::binary_cross_entropy(y_true, y_pred) << "\n";
    // } catch (const std::invalid_argument& e) {
    //     std::cout << "Caught error: " << e.what() << "\n";
    // }


    // // ---------------------
    // // LOGISTIC REGRESSION
    // // ---------------------
    // ml::LogisticRegression model;
    
    // // model.set_parameters(2.0, 1.0);
    // model.set_parameters(0.0, 0.0);

    // std::vector<double> x_values = {0.0, 1.0};
    // std::vector<double> y_values = {0.0, 1.0};

    // // Test: predict_proba
    // double x = 1.0;
    // std::cout << "\nLogistic Regression Predict Probability: " << model.predict_proba(x) << "\n";
    
    // // Test: predict_proba vector
    // std::cout << "\nLogistic Regression Predict Probabilities vector: ";
    // print_vector(model.predict_proba(x_values));

    // // Test: gradient
    // std::cout << "\nLogistic Regression Gradient:\n";
    // auto [grad_w, grad_b] = model.gradient(x_values, y_values);

    // std::cout << "xs: ";
    // print_vector(x_values);

    // std::cout << "ys: ";
    // print_vector(y_values);

    // std::cout << "Initial parameters:\n";
    // std::cout << "weight = " << model.weight() << "\n";
    // std::cout << "bias   = " << model.bias() << "\n";

    // std::cout << "Gradient weight: " << grad_w << "\n";
    // std::cout << "Gradient bias:   " << grad_b << "\n";

    // // Test: one GD step
    // double learning_rate = 0.01;
    // std::size_t epochs = 20;
    
    // std::cout << "\nGradient Descent Step:\n";
    // model.gradient_descent_step(x_values, y_values, learning_rate);
    
    // std::cout << "xs: ";
    // print_vector(x_values);

    // std::cout << "ys: ";
    // print_vector(y_values);

    // std::cout << "Updated parameters after one step:\n";
    // std::cout << "weight = " << model.weight() << "\n";
    // std::cout << "bias   = " << model.bias() << "\n";

    // std::cout << "Gradient weight: " << grad_w << "\n";
    // std::cout << "Gradient bias:   " << grad_b << "\n";

    // // Test: fit_batch
    // model.set_parameters(0.0, 0.0);
    
    // std::vector<double> y_pred_before = model.predict_proba(x_values);

    // std::cout << "Predicted probabilities before training: ";
    // print_vector(y_pred_before);

    // std::cout << "BCE before training: "
    //           << ml::common::binary_cross_entropy(y_values, y_pred_before) << "\n";
    
    // std::cout << "\nFit Batch:\n";
    // model.fit_batch(x_values, y_values, learning_rate, epochs);

    // std::cout << "xs: ";
    // print_vector(x_values);

    // std::cout << "ys: ";
    // print_vector(y_values);

    // std::cout << "Parameters after fit_batch:\n";
    // std::cout << "weight = " << model.weight() << "\n";
    // std::cout << "bias   = " << model.bias() << "\n";

    // auto [grad_w_after, grad_b_after] = model.gradient(x_values, y_values);
    // std::cout << "Gradient weight after fit_batch: " << grad_w_after << "\n";
    // std::cout << "Gradient bias after fit_batch:   " << grad_b_after << "\n";

    // std::vector<double> y_pred_after = model.predict_proba(x_values);
    // std::cout << "Predicted probabilities after training: ";
    // print_vector(y_pred_after);

    // std::cout << "BCE after training: "
    //           << ml::common::binary_cross_entropy(y_values, y_pred_after) << "\n";


    // ---------------------
    // LOGISTIC REGRESSION – Predictions
    // ---------------------
    ml::LogisticRegression model;
    model.set_parameters(0.0, 0.0);

    std::vector<double> x_values = {0.0, 1.0};
    std::vector<double> y_values = {0.0, 1.0};

    // Test 1: scalar prediction before training
    double x = 1.0;
    std::cout << "\nLogistic Regression Predictions:\n";
    std::cout << "\nScalar prediction before training:\n";
    std::cout << "x: " << x << "\n";
    std::cout << "predict_proba(x): " << model.predict_proba(x) << "\n";
    std::cout << "predict(x):       " << model.predict(x) << "\n";

    // Test 2: vector prediction before training
    std::cout << "\nVector prediction before training:\n";
    std::cout << "x_values: ";
    print_vector(x_values);

    std::vector<double> y_pred_before = model.predict_proba(x_values);
    std::vector<int> y_class_before = model.predict(x_values);

    std::cout << "predict_proba(x_values): ";
    print_vector(y_pred_before);
    std::cout << "predict(x_values):       ";
    print_vector(y_class_before);

    std::cout << "BCE before training: "
              << ml::common::binary_cross_entropy(y_values, y_pred_before) << "\n";

    // Test 3: train the model
    double learning_rate = 0.01;
    std::size_t epochs = 20;

    std::cout << "\nTraining with fit_batch:\n";
    std::cout << "learning_rate: " << learning_rate << "\n";
    std::cout << "epochs:        " << epochs << "\n";

    model.fit_batch(x_values, y_values, learning_rate, epochs);

    std::cout << "Parameters after training:\n";
    std::cout << "weight = " << model.weight() << "\n";
    std::cout << "bias   = " << model.bias() << "\n";

    // Test 4: scalar prediction after training
    std::cout << "\nScalar prediction after training:\n";
    std::cout << "x: " << x << "\n";
    std::cout << "predict_proba(x): " << model.predict_proba(x) << "\n";
    std::cout << "predict(x):       " << model.predict(x) << "\n";

    // Test 5: vector prediction after training
    std::cout << "\nVector prediction after training:\n";
    std::cout << "x_values: ";
    print_vector(x_values);

    std::vector<double> y_pred_after = model.predict_proba(x_values);
    std::vector<int> y_class_after = model.predict(x_values);

    std::cout << "predict_proba(x_values): ";
    print_vector(y_pred_after);
    std::cout << "predict(x_values):       ";
    print_vector(y_class_after);

    std::cout << "BCE after training: "
              << ml::common::binary_cross_entropy(y_values, y_pred_after) << "\n";

    // Test 6: empty vector should throw
    std::cout << "\nEmpty-vector prediction test:\n";
    try {
        std::vector<double> empty_values{};
        std::vector<int> empty_predictions = model.predict(empty_values);
        std::cout << "predict(empty_values): ";
        print_vector(empty_predictions);
    } catch (const std::invalid_argument& e) {
        std::cout << "Caught error: " << e.what() << "\n";
    }
    
    return 0;
}