#include "ml/linear_regression/linear_regression.hpp"
#include "ml/logistic_regression/logistic_regression.hpp"
#include "ml/knn/knn.hpp"
#include "ml/decision_tree/tree_node.hpp"
#include "ml/common/evaluation_metrics.hpp"

#include <iostream>
#include <vector>

namespace {

template <typename T>
void print_vector(const std::vector<T>& v) {
    std::cout << "[ ";
    for (std::size_t i = 0; i < v.size(); ++i) {
        std::cout << v[i];
        if (i + 1 < v.size()) {
            std::cout << ", ";
        }
    }
    std::cout << " ]\n";
}

void print_confusion_matrix(const ml::common::ConfusionMatrix& matrix) {
    std::cout << "TN: " << matrix.true_negatives << "\n";
    std::cout << "FP: " << matrix.false_positives << "\n";
    std::cout << "FN: " << matrix.false_negatives << "\n";
    std::cout << "TP: " << matrix.true_positives << "\n";
}

void print_tree_node(const ml::DecisionTreeNode& node) {
    std::cout << "is_leaf: " << std::boolalpha << node.is_leaf << "\n";
    std::cout << "prediction: " << node.prediction << "\n";
    std::cout << "left == nullptr: " << (node.left == nullptr) << "\n";
    std::cout << "right == nullptr: " << (node.right == nullptr) << "\n";
}

void run_linear_regression_demo() {
    ml::LinearRegression model;
    model.set_parameters(2.0, 1.0);

    std::vector<double> x_values = {1.0, 2.0, 3.0};
    std::vector<double> predictions = model.predict(x_values);

    std::cout << "\n== Linear Regression Demo ==\n";
    std::cout << "weight: " << model.weight() << "\n";
    std::cout << "bias:   " << model.bias() << "\n";
    std::cout << "x_values: ";
    print_vector(x_values);
    std::cout << "predictions: ";
    print_vector(predictions);
}

void run_logistic_regression_demo() {
    ml::LogisticRegression model;
    model.set_parameters(0.0, 0.0);

    std::vector<double> x_values = {0.0, 1.0};
    std::vector<double> y_values = {0.0, 1.0};

    model.fit_batch(x_values, y_values, 0.01, 20);

    std::vector<double> probabilities = model.predict_proba(x_values);
    std::vector<int> predictions = model.predict(x_values);

    std::cout << "\n== Logistic Regression Demo ==\n";
    std::cout << "weight: " << model.weight() << "\n";
    std::cout << "bias:   " << model.bias() << "\n";
    std::cout << "x_values: ";
    print_vector(x_values);
    std::cout << "probabilities: ";
    print_vector(probabilities);
    std::cout << "predictions: ";
    print_vector(predictions);
}

void run_evaluation_metrics_demo() {
    std::vector<int> y_true = {0, 1, 1, 0};
    std::vector<int> y_pred = {0, 1, 0, 0};

    std::cout << "\n== Evaluation Metrics Demo ==\n";
    std::cout << "y_true: ";
    print_vector(y_true);
    std::cout << "y_pred: ";
    print_vector(y_pred);
    std::cout << "accuracy:  " << ml::common::accuracy(y_true, y_pred) << "\n";
    std::cout << "precision: " << ml::common::precision(y_true, y_pred) << "\n";
    std::cout << "recall:    " << ml::common::recall(y_true, y_pred) << "\n";
    std::cout << "confusion matrix:\n";
    print_confusion_matrix(ml::common::confusion_matrix(y_true, y_pred));
}

void run_knn_demo() {
    std::vector<double> x_train = {1.0, 2.0, 5.0};
    std::vector<int> y_train = {0, 0, 1};
    int k = 1;

    ml::KNNClassifier model(x_train, y_train, k);

    std::vector<double> x_values = {1.2, 4.8};
    std::vector<int> predictions = model.predict(x_values);

    std::cout << "\n== k-NN Demo ==\n";
    std::cout << "x_train: ";
    print_vector(x_train);
    std::cout << "y_train: ";
    print_vector(y_train);
    std::cout << "k: " << k << "\n";
    std::cout << "x_values: ";
    print_vector(x_values);
    std::cout << "predictions: ";
    print_vector(predictions);
}

void run_decision_tree_demo() {
    std::cout << "\n== Decision Tree Demo ==\n";

    ml::DecisionTreeNode leaf_0;
    leaf_0.is_leaf = true;
    leaf_0.threshold = 0.0;
    leaf_0.prediction = 0;
    leaf_0.left = nullptr;
    leaf_0.right = nullptr;

    ml::DecisionTreeNode leaf_1;
    leaf_1.is_leaf = true;
    leaf_1.threshold = 0.0;
    leaf_1.prediction = 1;
    leaf_1.left = nullptr;
    leaf_1.right = nullptr;

    ml::DecisionTreeNode root;
    root.is_leaf = false;
    root.threshold = 4.0;
    root.prediction = -1;
    root.left = &leaf_0;
    root.right = &leaf_1;

    std::cout << "\nLeaf 0:\n";
    print_tree_node(leaf_0);

    std::cout << "\nLeaf 1:\n";
    print_tree_node(leaf_1);

    std::cout << "\nRoot:\n";
    print_tree_node(root);
    std::cout << "threshold: " << root.threshold << "\n";

    double x = 2.5;
    int prediction = (x < root.threshold) ? root.left->prediction : root.right->prediction;

    std::cout << "\nPrediction path test:\n";
    std::cout << "x: " << x << "\n";
    std::cout << "predicted class: " << prediction << "\n";

    x = 6.0;
    prediction = (x < root.threshold) ? root.left->prediction : root.right->prediction;

    std::cout << "\nPrediction path test:\n";
    std::cout << "x: " << x << "\n";
    std::cout << "predicted class: " << prediction << "\n";
}

}  // namespace

int main() {
    run_linear_regression_demo();
    run_logistic_regression_demo();
    run_evaluation_metrics_demo();
    run_knn_demo();
    run_decision_tree_demo();

    return 0;
}