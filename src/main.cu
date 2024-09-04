#include "../include/cuda_includes.cuh"
#include "../include/mat_update.cuh"
#include <stdio.h>
#include <stdlib.h>
#include <iostream>

using namespace std;

void print_mat(float* matrix, int m, int n) {
    for (int i=0; i<m; i++) {
        cout << "[ ";
        for (int j=0; j<n; j++) {
            cout << " " << matrix[i*n+j];
        }
        cout << "]\n";
    }
    cout << "\n";
}

bool compare_float(float x, float y, float epsilon = 0.01f){
    if(fabs(x - y) < epsilon)
        return true;
    return false;
}

void verify_result(float* CC, float* h_CC, int m, int n) {
    bool equal = true;
    for (int i=0; i<m; i++) {
        for (int j=0; j<n; j++) {
            if (!(compare_float(CC[i*n+j], h_CC[i*n+j]))) {
                equal = false;
            }
        }
    }

    if (equal) {
        cout << "CC and h_CC are equal." << endl << endl;
    } else {
        cout << "CC and h_CC element not equal." << endl;
    }
}

/*
Matrix Multiplication Golub and Van Loam Chapter 1 Section 6
Given C ( m x n), A (m x r), and B (r x n) compute CC = C + AB.
*/
int main(int argc, char const *argv[]) {
    int m, n, r;
    srand(3333);
    printf("Please type in m n and r:\n");
    scanf("%d %d %d", &m, &n, &r);

    float* A;
    float* B;
    float* C;
    float* CC;
    float* h_CC;
    size_t A_size = sizeof(float)*m*r;
    size_t B_size = sizeof(float)*r*n;
    size_t C_size = sizeof(float)*m*n;
    size_t CC_size = sizeof(float)*m*n;
    size_t h_CC_size = CC_size;

    cudaMallocManaged((void **) &A, A_size);
    cudaMallocManaged((void **) &B, B_size);
    cudaMallocManaged((void **) &C, C_size);
    cudaMallocManaged((void **) &CC, CC_size);
    h_CC = (float*) malloc(h_CC_size);

    // random initialize matrix A
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < r; j++) {
            A[i*r + j] = float(rand());
        }
    }

    // random initialize matrix B
    for (int i = 0; i < r; i++) {
        for (int j = 0; j < n; j++) {
            B[i*n + j] = float(rand());
        }
    }

    // random initialize matrix C and CC
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            C[i*n + j] = float(rand());
            CC[i*n + j] = 0;
        }
    }

    // Print matrices
    cout << "Matrix A:\n";
    print_mat(A, m, r);
    cout << "Matrix B:\n";
    print_mat(B,r,n);
    cout << "Matrix C:\n";
    print_mat(C,m,n);
    cout << "Matrix CC:\n";
    print_mat(CC,m,n);

    //Block dimension is directly from block_size
    dim3 THREADS(BLOCK_SIZE, BLOCK_SIZE);
    //Grid dimension is found by dividing matrix dimension by block_size
    dim3 BLOCKS((m+BLOCK_SIZE-1)/BLOCK_SIZE, (n+BLOCK_SIZE-1)/BLOCK_SIZE);

    //CPU mat update
    host_mat_update(A,B,C,h_CC,m,n,r);

    //Standard mat update
    cudaDeviceSynchronize();
    tiled_mat_update<<<THREADS, BLOCKS>>>(A, B, C, CC, m, n , r);
    cudaDeviceSynchronize();

    cout << "Host Matrix CC:\n";
    print_mat(h_CC, m, n);
    cout << "Device Matrix CC:\n";
    print_mat(CC, m, n);
    verify_result(CC, h_CC, m, n);

    // Reset GPU CC matrix
    for (int i=0; i<m; i++){
        for (int j=0; j<n; j++) {
            CC[i*n+j] = 0;
        }
    }

    cudaDeviceSynchronize();
    tiled_mat_update<<<THREADS, BLOCKS>>>(A, B, C, CC, m, n , r);
    cudaDeviceSynchronize();

    cout << "Host Matrix CC:\n";
    print_mat(h_CC, m, n);
    cout << "Device Matrix CC:\n";
    print_mat(CC, m, n);
    verify_result(CC, h_CC, m, n);

    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    cudaFree(CC);
    free(h_CC);
}