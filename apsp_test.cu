#include "apsp.cu"
#include <chrono>
#include <fstream>
#include <iostream>
#include <random>
#include <vector>
using namespace std;

#define n (1024)
int d[n][n], res1[n][n], res2[n][n];

int main() {
    std::random_device rd;
    std::mt19937 gen(rd());

    apsp::FloydWarshall wf(n);

    std::chrono::high_resolution_clock::time_point t1, t2;
    std::chrono::duration<double> time_span;

    std::fstream in("in.txt", std::ios::out);
    std::fstream out1("out1.txt", std::ios::out);
    std::fstream out2("out2.txt", std::ios::out);

    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            d[i][j] = i == j ? 0 : gen() % 1000 + 1;
            in << d[i][j] << " ";
        }
        in << std::endl;
    }

    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) { wf.SetDist(i, j, d[i][j]); }
    }

    cudaDeviceSynchronize();
    t1 = std::chrono::high_resolution_clock::now();
    wf.DeviceCalc();
    cudaDeviceSynchronize();
    t2 = std::chrono::high_resolution_clock::now();
    time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    std::cout << "GPU: Time: " << time_span.count() << "s" << std::endl;

    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            res1[i][j] = wf.GetDist(i, j);
            out1 << res1[i][j] << " ";
        }
        out1 << std::endl;
    }

    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) { wf.SetDist(i, j, d[i][j]); }
    }

    t1 = std::chrono::high_resolution_clock::now();
    wf.Calc();
    t2 = std::chrono::high_resolution_clock::now();
    time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    std::cout << "CPU: Time: " << time_span.count() << "s" << std::endl;

    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            res2[i][j] = wf.GetDist(i, j);
            out2 << res2[i][j] << " ";
        }
        out2 << std::endl;
    }

    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            if (res1[i][j] != res2[i][j]) {
                std::cout << "Error: res[" << i << "][" << j << "]: " << res1[i][j] << " / "
                          << res2[i][j] << std::endl;
                return 1;
            }
        }
    }
    std::cout << "OK" << std::endl;
}
