#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <math.h>

#define BLOCK_SIZE 16
#define N 256

__global__ void heat_transfer(float *u, float *u_new, float kx, float ky, float kz, float dt, int nx, int ny, int nz)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= 1 && i < nx-1 && j >= 1 && j < ny-1 && k >= 1 && k < nz-1) {
        u_new[i*ny*nz + j*nz + k] = u[i*ny*nz + j*nz + k] + kx*(u[(i+1)*ny*nz + j*nz + k] + u[(i-1)*ny*nz + j*nz + k] - 2*u[i*ny*nz + j*nz + k]) + ky*(u[i*ny*nz + (j+1)*nz + k] + u[i*ny*nz + (j-1)*nz + k] - 2*u[i*ny*nz + j*nz + k]) + kz*(u[i*ny*nz + j*nz + (k+1)] + u[i*ny*nz + j*nz + (k-1)] - 2*u[i*ny*nz + j*nz + k]);
        u_new[i*ny*nz + j*nz + k] = u_new[i*ny*nz + j*nz + k]*dt;
    }
}

int main(void)
{
    float *u, *u_new;
    float *d_u, *d_u_new;
    float kx = 1.0, ky = 1.0, kz = 1.0;   //For isotropic material, all are same.
    float dt = 0.01;
    int nx = N, ny = N, nz = N;
    int n = nx*ny*nz;
    int i, j, k, t;

    u = (float*) malloc(n*sizeof(float));
    u_new = (float*) malloc(n*sizeof(float));

    cudaMalloc((void**)&d_u, n*sizeof(float));
    cudaMalloc((void**)&d_u_new, n*sizeof(float));

    // set initial conditions
    for (i=0; i<n; i++) {
        u[i] = 0.0;
        u_new[i] = 0.0;
    }

    // set boundary conditions
    for (i=0; i<nx; i++) {
        for (j=0; j<ny; j++) {
            for (k=0; k<nz; k++) {
                if (i==0 || i==nx-1 || j==0 || j==ny-1 || k==0 || k==nz-1) {
                    u[i*ny*nz + j*nz + k] = 1.0;
                    u_new[i*ny*nz + j*nz + k] = 1.0;
                }
            }
        }
    }

    cudaMemcpy(d_u, u, n*sizeof(float), cudaMemcpyHostToDevice);

    dim3 dimGrid((nx-1)/BLOCK_SIZE+1, (ny-1)/BLOCK_SIZE+1, (nz-1)/BLOCK_SIZE+1);
    dim3 dimBlock(BLOCK, BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);

for (t=0; t<100; t++) {
    heat_transfer<<<dimGrid, dimBlock>>>(d_u, d_u_new, kx, ky, kz, dt, nx, ny, nz);
    cudaMemcpy(u, d_u_new, n*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(d_u, u, n*sizeof(float), cudaMemcpyHostToDevice);
}

cudaFree(d_u);
cudaFree(d_u_new);
free(u);
free(u_new);

return 0;
}
