#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>

#define NX 256
#define NY 256
#define TILE_SIZE 32

// Device function to initialize the velocity and pressure fields
__global__ void init_fields(float *u, float *v, float *p, int nx, int ny)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = j * nx + i;
    if (i < nx && j < ny) {
        u[idx] = 0.0;
        v[idx] = 0.0;
        p[idx] = 0.0;
    }
}


// Device function to compute the velocity and pressure fields
__global__ void compute_fields(float *u, float *v, float *p, float *u_new, float *v_new, float *p_new, float dx, float dy, float dt, float nu, int nx, int ny)
{
    __shared__ float su[TILE_SIZE][TILE_SIZE];
    __shared__ float sv[TILE_SIZE][TILE_SIZE];
    __shared__ float sp[TILE_SIZE][TILE_SIZE];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = j * nx + i;
    if (i > 0 && i < nx - 1 && j > 0 && j < ny - 1) {
        // Compute x-velocity
        float uxx = (u[(j + 1) * nx + i] - 2.0 * u[idx] + u[(j - 1) * nx + i]) / (dx * dx);
        float uyy = (u[j * nx + i + 1] - 2.0 * u[idx] + u[j * nx + i - 1]) / (dy * dy);
        float uxy = (u[(j + 1) * nx + i + 1] - u[(j + 1) * nx + i - 1] - u[(j - 1) * nx + i + 1] + u[(j - 1) * nx + i - 1]) / (4.0 * dx * dy);
        float uadvx = -u[idx] * (u[(j + 1) * nx + i] - u[(j - 1) * nx + i]) / (2.0 * dx) - v[idx] * (u[j * nx + i + 1] - u[j * nx + i - 1]) / (2.0 * dy);
        float udiffx = nu * (uxx + uyy) + uxy;
        u_new[idx] = u[idx] + dt * (udiffx + uadvx);

        // Compute y-velocity
        float vxx = (v[(j + 1) * nx + i] - 2.0 * v[idx] + v[(j - 1) * nx + i]) / (dx * dx);
        float vyy = (v[j * nx + i + 1] - 2.0 * v[idx] + v[j * nx + i - 1]) / (dy * dy);
        float vxy = (v[(j + 1) * nx + i + 1] - v[(j + 1) * nx + i - 1] - v[(j - 1) * nx + i + 1] + v[(j - 1) * nx + i - 1]) / (4.0 * dx * dy);
        float vadvy = -u[idx] * (v[(j + 1) * nx + i] - v[(j - 1) * nx + i]) / (2.0 * dx) - v[idx] * (v[j * nx + i + 1] - v[j * nx + i - 1]) / (2.0 * dy);
        float udiffy = nu * (vxx + vyy) + vxy;
        v_new[idx] = v[idx] + dt * (udiffy + vadvy);

        // Compute pressure
        float pxx = (p[(j + 1) * nx + i] - 2.0 * p[idx] + p[(j - 1) * nx + i]) / (dx * dx);
        float pyy = (p[j * nx + i + 1] - 2.0 * p[idx] + p[j * nx + i - 1]) / (dy * dy);
        float pxy = (p[(j + 1) * nx + i + 1] - p[(j + 1) * nx + i - 1] - p[(j - 1) * nx + i + 1] + p[(j - 1) * nx + i - 1]) / (4.0 * dx * dy);
        float divu = (u[(j + 1) * nx + i] - u[(j - 1) * nx + i]) / (2.0 * dx) + (v[j * nx + i + 1] - v[j * nx + i - 1]) / (2.0 * dy);
        float pnew = p[idx] + dt * (-divu - pxx - pyy);

        // Update fields
        u_new[idx] = u[idx] + dt * (udiffx + uadvx);
        v_new[idx] = v[idx] + dt * (udiffy + vadvy);
        p_new[idx] = pnew;
  }
}


int main()
{
float *u, *v, *p;
float *u_new, *v_new, *p_new;
float dx = 1.0 / NX;
float dy = 1.0 / NY;
float dt = 0.001;
float nu = 0.01;
// Allocate memory on host and device
size_t size = NX * NY * sizeof(float);
u = (float*)malloc(size);
v = (float*)malloc(size);
p = (float*)malloc(size);
u_new = (float*)malloc(size);
v_new = (float*)malloc(size);
p_new = (float*)malloc(size);
cudaMalloc((void**)&u, size);
cudaMalloc((void**)&v, size);
cudaMalloc((void**)&p, size);
cudaMalloc((void**)&u_new, size);
cudaMalloc((void**)&v_new, size);
cudaMalloc((void**)&p_new, size);

// Initialize fields
dim3 blocks(NX / TILE_SIZE, NY / TILE_SIZE);
dim3 threads(TILE_SIZE, TILE_SIZE);
init_fields<<<blocks, threads>>>(u,v,p,NX,NY);

// Main time loop
for (int n = 0; n < NT; n++)
{
    // Update velocity and pressure fields
    dim3 blocks((NX - 2) / TILE_SIZE, (NY - 2) / TILE_SIZE);
    dim3 threads(TILE_SIZE, TILE_SIZE);
    compute_fields<<<blocks, threads>>>(u, v, p, u_new, v_new, p_new, dx, dy, dt, NX, NY);
  
    // Copy new fields back to old fields
    cudaMemcpy(u, u_new, size, cudaMemcpyDeviceToDevice);
    cudaMemcpy(v, v_new, size, cudaMemcpyDeviceToDevice);
    cudaMemcpy(p, p_new, size, cudaMemcpyDeviceToDevice);
}

// Free memory
cudaFree(u);
cudaFree(v);
cudaFree(p);
cudaFree(u_new);
cudaFree(v_new);
cudaFree(p_new);
free(u);
free(v);
free(p);
free(u_new);
free(v_new);
free(p_new);

return 0;
  
}


