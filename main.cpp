#include <iostream>
#include <cstring>
#include "radiator_cpu.h"

int main(int argc, char **argv) {
    int n = 32, m = 32, p = 10;
    bool do_avg = false;

    // Parse command-line arguments
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            n = std::atoi(argv[++i]);
        } else if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            m = std::atoi(argv[++i]);
        } else if (strcmp(argv[i], "-p") == 0 && i + 1 < argc) {
            p = std::atoi(argv[++i]);
        } else if (strcmp(argv[i], "-a") == 0) {
            do_avg = true;
        } else {
            std::cerr << "Unknown option " << argv[i] << "\n";
            return 1;
        }
    }

    // Allocate memory
    float *mat  = new float[n * m];
    float *next = new float[n * m];

    initialize(mat, next, n, m);

    // Perform p iterations
    for (int iter = 0; iter < p; ++iter) {
        propagate_step(mat, next, n, m);
        std::swap(mat, next);
    }

    // Compute averages and print
    float *avgs = new float[n];
    compute_averages(mat, avgs, n, m);

    // If -a is specified, compute and print
    if (do_avg) {
        std::cout << "Row averages:\n";
        for (int i = 0; i < n; ++i)
            std::cout << "  row " << i << ": " << avgs[i] << "\n";
    } else {
        float total = 0.0f;
        for (int i = 0; i < n; ++i)
            total += avgs[i];
        float avg_of_avgs = total / n;
        std::cout << "Completed " << p << " iterations on " << n << "x" << m << " matrix.\n";
        std::cout << "Overall average temperature: " << avg_of_avgs << "\n";
    }
    
    delete[] avgs;

    delete[] mat;
    delete[] next;
    return 0;
}
