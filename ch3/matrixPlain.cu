#include <iostream>
#include <cuda_runtime.h>
#include "timer.hpp"

#define BLOCK_SIZE 16


struct Matrix {
    int width;
    int height;
    float *elements;

    Matrix(int width, int height) : width(width), height(height) {
        size_t size = width * height * sizeof(float);
        void *ptr = ::operator new (size);
        std::memset(ptr, 0, size);
        elements = static_cast<float*>(ptr);
    }


    float &get(int i, int j) {
        return elements[i * width + j];
    }
    float &get(int i, int j) const {
        return elements[i * width + j];
    }

    Matrix &random() {
        for (int i = 0; i < height; ++i) {
            for (int j = 0; j < width; ++j) {
                this->get(i, j) = static_cast<float>(std::rand() % 100);
            }
        }
        return *this;
    }

    ~Matrix() {
        delete elements;
    }
};

__global__ void MatMulKernel(const Matrix &, const Matrix &, Matrix &);


void CpuMatMul(const Matrix &A, const Matrix &B, Matrix &C) {
    for (int i = 0; i < A.height; ++i) {
        for (int j = 0; j < B.width; ++j) {
            for (int k = 0; k < A.width; ++k) {
                C.get(i, j) += A.get(i, k) * B.get(k, j);
            }
        }
    }
}

void MatMul(const Matrix &A, const Matrix &B, Matrix &C) {
    Matrix d_A(A.width, A.height);
    size_t size = A.width * A.height * sizeof(float);
    cudaMalloc(&d_A.elements, size);
    cudaMemcpy(d_A.elements, A.elements, size, cudaMemcpyHostToDevice);

    Matrix d_B(B.width, B.height);
    size = B.width * B.height * sizeof(float);
    cudaMalloc(&d_B.elements, size);
    cudaMemcpy(d_B.elements, B.elements, size, cudaMemcpyHostToDevice);

    Matrix d_C(C.width, C.height);
    size = C.width * C.height * sizeof(float);
    cudaMalloc(&d_C.elements, size);

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((B.width + dimBlock.x - 1) / dimBlock.x, (A.height + dimBlock.y - 1) / dimBlock.y);
    MatMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);

    cudaMemcpy(C.elements, d_C.elements, size, cudaMemcpyDeviceToHost);
    cudaFree(d_A.elements);
    cudaFree(d_B.elements);
    cudaFree(d_C.elements);
}

int main() {
    int A_width = 512;
    int A_height = 512;
    int B_width = 1000;
    auto &step_timer = Timer::Global();
    Matrix A(A_width, A_height);
    A.random();
    Matrix B(B_width, A_width);
    B.random();
    Matrix C(B_width, A_height);
    Matrix D(B_width, A_height);

    step_timer.start("cpu");
    CpuMatMul(A, B, C);
    step_timer.log("cpu");

    step_timer.start("gpu-plain");
    MatMul(A, B, D);
    step_timer.log("gpu-plain");
    
    std::cout << step_timer.to_string(Timer::Unit::kMicroSecond) << std::endl;

    // float gap = 0.;
    // for (int i = 0; i < A.height; ++i) {
    //     for (int j = 0; j < B.width; ++j) {
    //         gap += abs(C.get(i, j) - D.get(i, j));
    //     }
    // }
    // if (gap > 1e-5) {
    //     std::cout << "The result of cpu and gpu_plain is different." << std::endl;
    // }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Cuda error occur." << std::endl;
        exit(EXIT_FAILURE);
    } else {
        std::cout << "done" << std::endl;
        exit(EXIT_SUCCESS);
    }
}

__global__ void MatMulKernel(const Matrix &A, const Matrix &B, Matrix &C) {
    float Cvalue = 0;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= A.height || col >= B.width)
        return;
    for (int e = 0; e < A.width; ++e)
        Cvalue += A.elements[row * A.width + e] * B.elements[e * B.width + col];
    C.elements[row * C.width + col] = Cvalue;
}