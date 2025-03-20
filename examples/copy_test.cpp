#include <iostream>
#include <vector>
#include <chrono>
#include "Backend.hpp"
#include "Layer.hpp"
#include "Module.hpp"
#include "Tensor.hpp"
#include "Types.hpp"
#include "backends/cpu/CPUBackend.hpp"

using namespace std;
using namespace mllm;
using namespace std::chrono;

void benchmark_tensor_transfer(int num_trials, vector<int> tensor_shape) {
    cout << "Benchmarking Tensor Transfer Performance (CPU <-> QNN)" << endl;
    
    Module::initBackend(MLLM_CPU);
    Module::initBackend(MLLM_QNN);

    int bs = tensor_shape[0];
    int num_heads = tensor_shape[1];
    int seq_len = tensor_shape[2];
    int hidden_dim = tensor_shape[3];
    auto x = Tensor(bs, num_heads, seq_len, hidden_dim, MLLM_CPU); 

    double cpu_to_qnn_total = 0.0;
    for (int i = 0; i < num_trials; ++i) {
        auto start = high_resolution_clock::now();
        Tensor qnn_x = Tensor::toQNN({x})[0];  // CPU -> QNN
        auto end = high_resolution_clock::now();
        cpu_to_qnn_total += duration<double, std::micro>(end - start).count();
    }
    
    double qnn_to_cpu_total = 0.0;
    Tensor qnn_x = Tensor::toQNN({x})[0];
    for (int i = 0; i < num_trials; ++i) {
        auto start = high_resolution_clock::now();
        Tensor cpu_x = Tensor::toCPU({qnn_x})[0];  // QNN -> CPU
        auto end = high_resolution_clock::now();
        qnn_to_cpu_total += duration<double, std::micro>(end - start).count();
    }

    double avg_cpu_to_qnn = cpu_to_qnn_total / num_trials;
    double avg_qnn_to_cpu = qnn_to_cpu_total / num_trials;

    cout << "Tensor Shape: [" << tensor_shape[0] << ", " << tensor_shape[1] << ", " << tensor_shape[2] << "]" << endl;
    cout << "Average CPU -> QNN transfer time: " << avg_cpu_to_qnn << " us" << endl;
    cout << "Average QNN -> CPU transfer time: " << avg_qnn_to_cpu << " us" << endl;
}

int main() {
    int num_trials = 10;

    vector<vector<int>> tensor_shapes = {
        {1, 32, 256, 128},
        {1, 32, 512, 128},
        {1, 32, 1024, 128}
    };

    for (auto& shape : tensor_shapes) {
        benchmark_tensor_transfer(num_trials, shape);
    }

    return 0;
}