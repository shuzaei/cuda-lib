#include <cstdlib>
#pragma once

namespace apsp {
    typedef unsigned long size_t;
    typedef int T;
#define BS (12)
#define INF (INT_MAX / 2)

    __global__ void Calc_(int n, T *deviceInput) {
        size_t i = blockIdx.x * blockDim.x + threadIdx.x;
        size_t j = blockIdx.y * blockDim.y + threadIdx.y;

        size_t bi = blockIdx.x;
        size_t bj = blockIdx.y;

        size_t ti = threadIdx.x;
        size_t tj = threadIdx.y;

        T res = deviceInput[i * n + j];

        for (size_t b1 = bi * BS * n, b2 = bj * BS; b2 < n * n; b1 += BS, b2 += BS * n) {

            __shared__ T sm1[BS][BS], sm2[BS][BS];

            sm1[ti][tj] = deviceInput[b1 + ti * n + tj];
            sm2[ti][tj] = deviceInput[b2 + ti * n + tj];

            __syncthreads();

            for (size_t k = 0; k < BS; k++) {
                if (sm1[ti][k] + sm2[k][tj] < res) { res = sm1[ti][k] + sm2[k][tj]; }
            }

            __syncthreads();
        }

        deviceInput[i * n + j] = res;
    }

    class FloydWarshall {
        private:
        size_t n;
        T **distance, *deviceInput;

        public:
        FloydWarshall(size_t n_) {
            n = (n_ + BS - 1) / BS * BS;
            distance = (T **) malloc(n * sizeof(T *));
            distance[0] = (T *) malloc(n * n * sizeof(T));

            cudaMalloc((void **) &deviceInput, n * n * sizeof(T));

            for (size_t i = 0; i < n; i++) {
                distance[i] = distance[0] + i * n;
                for (size_t j = 0; j < n; j++) { distance[i][j] = INF; }
            }
        }
        ~FloydWarshall() {
            free(distance[0]);
            free(distance);
            cudaFree(deviceInput);
        }

        void SetDist(size_t i, size_t j, T c) { distance[i][j] = c; }
        T GetDist(size_t i, size_t j) { return distance[i][j]; }
        void Calc() {
            for (size_t k = 0; k < n; k++) {
                for (size_t i = 0; i < n; i++) {
                    for (size_t j = 0; j < n; j++) {
                        if (distance[i][k] + distance[k][j] < distance[i][j]) {
                            distance[i][j] = distance[i][k] + distance[k][j];
                        }
                    }
                }
            }
        }
        void DeviceCalc() {
            cudaMemcpy(deviceInput, (void *) distance[0], n * n * sizeof(T), cudaMemcpyDefault);
            for (size_t r = 1; r < n; r <<= 1) {
                Calc_<<<dim3((n + BS - 1) / BS, (n + BS - 1) / BS), dim3(BS, BS)>>>(n, deviceInput);
            }
            cudaMemcpy((void *) distance[0], deviceInput, n * n * sizeof(T), cudaMemcpyDefault);
            for (size_t i = 0; i < n; i++) { distance[i] = distance[0] + i * n; }
        }
    };

#undef BS
#undef inf
}; // namespace apsp
