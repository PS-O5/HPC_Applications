#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>

#define BLOCK_SIZE 16

__global__ void jacobi_kernel(double *u, double *f, int N, double h, double *delta)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (i > 0 && i < N-1 && j > 0 && j < N-1)
    {
        double u_new = (f[i*N+j]*h*h + u[(i-1)*N+j] + u[(i+1)*N+j] + u[i*N+j-1] + u[i*N+j+1])/4.0;
        delta[i*N+j] = u_new - u[i*N+j];
        u[i*N+j] = u_new;
    }
}

void jacobi_gpu(double *u, double *f, int N, double h, double tol, int max_iter)
{
    double *delta;
    cudaMalloc(&delta, N*N*sizeof(double));
    
    int iter = 0;
    double resid = 2*tol;
    
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((N-2+BLOCK_SIZE-1)/BLOCK_SIZE, (N-2+BLOCK_SIZE-1)/BLOCK_SIZE);
    
    while (iter < max_iter && resid > tol)
    {
        jacobi_kernel<<<grid, block>>>(u, f, N, h, delta);
        cudaDeviceSynchronize();
        
        double sum = 0.0;
        for (int i=1; i<N-1; i++)
        {
            for (int j=1; j<N-1; j++)
            {
                sum += delta[i*N+j]*delta[i*N+j];
            }
        }
        resid = sqrt(sum);
        
        iter++;
    }
    
    cudaFree(delta);
}

int main()
{
    int N = 256;
    double L = 1.0;
    double h = L/(N-1);
    double tol = 1e-5;
    int max_iter = 1000;
    
    double *u, *f;
    cudaMallocManaged(&u, N*N*sizeof(double));
    cudaMallocManaged(&f, N*N*sizeof(double));
    
    for (int i=0; i<N; i++)
    {
        for (int j=0; j<N; j++)
        {
            if (i==0 || i==N-1 || j==0 || j==N-1)
            {
                u[i*N+j] = 0.0; // Dirichlet boundary condition
            }
            else
            {
                u[i*N+j] = 1.0; // initial guess
            }
            f[i*N+j] = sin(M_PI*i*h)*sin(M_PI*j*h); // source term
        }
    }
    
    jacobi_gpu(u, f, N, h, tol, max_iter);
    
    // print solution
    for (int i=0; i<N; i++)
    {
        for (int j=0; j<N; j++)
        {
            printf("%f ", u[i*N+j]);
        }
        printf("\n");
    }
    
    cudaFree(u);
    cudaFree(f);
    
    return 0;
}
