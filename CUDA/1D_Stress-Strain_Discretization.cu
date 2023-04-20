#include <stdio.h>

#define N 1000000 // number of elements
#define BLOCK_SIZE 512 // number of threads in a block

// material properties
#define E 1e9 // Young's modulus (Pa)
#define nu 0.3 // Poisson's ratio

__global__ void compute_stress_and_strain(double *d_stress, double *d_strain, double *d_displacement) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    double r = 0.01; // radius of the bar (m)
    double L = 1.0; // length of the bar (m)

    double dx = L / N; // element length
    double A = M_PI * r * r; // cross-sectional area

    double eps, sig; // strain and stress

    if (tid < N) {
        // compute strain
        eps = d_displacement[tid + 1] - d_displacement[tid];
        eps /= dx;

        // compute stress
        sig = E / (1.0 - nu * nu) * (eps - nu * (d_displacement[tid + 1] - d_displacement[tid - 1]) / (2 * dx));

        // store stress and strain
        d_stress[tid] = sig;
        d_strain[tid] = eps;
    }
}

int main() {
    double *d_displacement, *d_stress, *d_strain;
    double *h_displacement, *h_stress, *h_strain;

    size_t size = sizeof(double) * (N + 1);
    cudaMalloc(&d_displacement, size);
    cudaMalloc(&d_stress, size);
    cudaMalloc(&d_strain, size);

    h_displacement = (double*) malloc(size);
    h_stress = (double*) malloc(size);
    h_strain = (double*) malloc(size);

    // initialize displacement
    for (int i = 0; i <= N; i++) {
        h_displacement[i] = i * 1e-6; // displacement in meters
    }

    cudaMemcpy(d_displacement, h_displacement, size, cudaMemcpyHostToDevice);

    int num_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    compute_stress_and_strain<<<num_blocks, BLOCK_SIZE>>>(d_stress, d_strain, d_displacement);

    cudaMemcpy(h_stress, d_stress, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_strain, d_strain, size, cudaMemcpyDeviceToHost);

    // print stress and strain
    for (int i = 0; i < N; i++) {
        printf("Element %d: stress = %f Pa, strain = %f\n", i, h_stress[i], h_strain[i]);
    }

    cudaFree(d_displacement);
    cudaFree(d_stress);
    cudaFree(d_strain);

    free(h_displacement);
    free(h_stress);
    free(h_strain);

    return 0;
}
