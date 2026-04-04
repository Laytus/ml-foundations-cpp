#include "ml/linear_regression/linear_regression.hpp"
#include "ml/common/vector_ops.hpp"
#include "ml/common/csv_loader.hpp"
#include "ml/common/data_split.hpp"

#include <iostream>
#include <vector>

void print_vector(const std::vector<double>& v) {
    std::cout <<"[ ";
    for (double value : v) {
        std::cout << value << " ";
    }
    std::cout << "]\n";
}

void print_dataset(const ml::common::Dataset& dataset, const std::string& name) {
    std::cout << name << " features:\n";
    for (const auto& row : dataset.features) {
        print_vector(row);
    }
    
    std::cout << name << " targets:\n";
    print_vector(dataset.targets);
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

    try {
        ml::common::Dataset dataset = ml::common::load_csv("data/test_linear.csv");

        auto [train_dataset, test_dataset] = ml::common::train_test_split(dataset, 0.75);

        std::cout << "Full dataset loaded successfully.\n\n";

        print_dataset(train_dataset, "Train");
        std::cout << "\n";
        print_dataset(test_dataset, "Test");
    } catch (const std::exception& e) {
        std::cout << "Error: " << e.what() << "\n";
    }

    return 0;
}