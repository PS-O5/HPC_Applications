#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define NX 128 // number of grid points in x direction
#define NY 128 // number of grid points in y direction
#define L 1.0 // length of domain
#define H 1.0 // height of domain
#define DX (L / (NX - 1)) // grid spacing in x direction
#define DY (H / (NY - 1)) // grid spacing in y direction
#define DT (0.0001) // time step size
#define T_FINAL (0.01) // final time
#define RE (100.0) // Reynolds number
#define U_IN (1.0) // inlet velocity
#define EPS (1e-5) // convergence criterion

__global__ void init(double *d_u, double *d_v, double *d_p) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < NX && idy < NY) {
        d_u[idx * NY + idy] = 0.0;
        d_v[idx * NY + idy] = 0.0;
        d_p[idx * NY + idy] = 0.0;
    }
}

__global__ void apply_bc(double *d_u, double *d_v) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx == 0 || idx == NX - 1 || idy == 0 || idy == NY - 1) {
        d_u[idx * NY + idy] = 0.0;
        d_v[idx * NY + idy] = 0.0;
    }
}

__global__ void compute_rhs(double *d_u, double *d_v, double *d_f, double *d_g, double *d_rhs) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx > 0 && idx < NX - 1 && idy > 0 && idy < NY - 1) {
        double u_n = d_u[idx * NY + idy];
        double u_s = d_u[(idx - 1) * NY + idy];
        double u_e = d_u[idx * NY + idy + 1];
        double u_w = d_u[idx * NY + idy - 1];
        double v_n = d_v[idx * NY + idy];
        double v_s = d_v[(idx - 1) * NY + idy];
        double v_e = d_v[idx * NY + idy + 1];
        double v_w = d_v[idx * NY + idy - 1];
        double f_c = d_f[idx * NY + idy];
        double g_c = d_g[idx * NY + idy];

        double du_dx = (u_e - u_w) / (2 * DX);
        double du_dy = (u_n - u_s) / (2 * DY);
        double dv_dx = (v_e - v_w) / (2 * DX);
        double dv_dy = (v_n - v_s) / (2 * DY);
        double d2u_dx2 = (u_e - 2 * u_n + u_w) / (DX * DX);
        double d2u_dy2 = (u_n - 2 * u_c + u_s) / (DY * DY);
        double d2v_dx2 = (v_e - 2 * v_c + v_w) / (DX * DX);
        double d2v_dy2 = (v_n - 2 * v_c + v_s) / (DY * DY);
        double viscous_x = (1.0 / RE) * (d2u_dx2 + d2u_dy2);
        double viscous_y = (1.0 / RE) * (d2v_dx2 + d2v_dy2);
        double convective_x = u_c * du_dx + v_c * du_dy;
        double convective_y = u_c * dv_dx + v_c * dv_dy;

        double rhs_u = viscous_x - convective_x;
        double rhs_v = viscous_y - convective_y;

        d_rhs[idx * NY + idy] = (1.0 / DT) * (f_c - u_n) + rhs_u;
        d_rhs[(NX * NY) + idx * NY + idy] = (1.0 / DT) * (g_c - v_n) + rhs_v;
    }
}

global void compute_p(double *d_p, double *d_rhs) {
int idx = blockIdx.x * blockDim.x + threadIdx.x;
int idy = blockIdx.y * blockDim.y + threadIdx.y;
if (idx > 0 && idx < NX - 1 && idy > 0 && idy < NY - 1) {
    double p_c = d_p[idx * NY + idy];
    double p_n = d_p[(idx + 1) * NY + idy];
    double p_s = d_p[(idx - 1) * NY + idy];
    double p_e = d_p[idx * NY + idy + 1];
    double p_w = d_p[idx * NY + idy - 1];
    double rhs_c = d_rhs[idx * NY + idy];
    double rhs_e = d_rhs[idx * NY + idy + 1];
    double rhs_w = d_rhs[idx * NY + idy - 1];
    double rhs_n = d_rhs[(NX * NY) + idx * NY + idy];
    double rhs_s = d_rhs[(NX * NY) + (idx - 1) * NY + idy];

    d_p[idx * NY + idy] = (1.0 / (2.0 / (DX * DX) + 2.0 / (DY * DY))) * ((p_e + p_w) / (DX * DX) + (p_n + p_s) / (DY * DY) - rhs_c);
    }
}

global void update_u(double *d_u, double *d_f, double *d_p) {
int idx = blockIdx.x * blockDim.x + threadIdx.x;
int idy = blockIdx.y * blockDim.y + threadIdx.y;
if (idx > 0 && idx < NX - 1 && idy > 0 && idy < NY - 1) {
    double p_e = d_p[idx * NY + idy + 1];
    double p_w = d_p[idx * NY + idy - 1];
    double u_c = d_u[idx * NY + idy];
    double f_c = d_f[idx * NY + idy];

    d_u[idx * NY + idy] = f_c - DT * (p_e - p_w) / (2 * DX);
    }
}

global void update_v(double *d_v, double*d_g, double *d_p) {
int idx = blockIdx.x * blockDim.x + threadIdx.x;
int idy = blockIdx.y * blockDim.y + threadIdx.y;
if (idx > 0 && idx < NX - 1 && idy > 0 && idy < NY - 1) {
    double p_n = d_p[(idx + 1) * NY + idy];
    double p_s = d_p[(idx - 1) * NY + idy];
    double v_c = d_v[idx * NY + idy];
    double g_c = d_g[idx * NY + idy];

    d_v[idx * NY + idy] = g_c - DT * (p_n - p_s) / (2 * DY);
    }
}

int main() {
double *h_u, *h_v, *h_f, *h_g, *h_p, *h_rhs;
double *d_u, *d_v, *d_f, *d_g, *d_p, *d_rhs;
int size = NX * NY * sizeof(double);

h_u = (double *) malloc(size);
h_v = (double *) malloc(size);
h_f = (double *) malloc(size);
h_g = (double *) malloc(size);
h_p = (double *) malloc(size);
h_rhs = (double *) malloc(2 * size);

cudaMalloc((void **) &d_u, size);
cudaMalloc((void **) &d_v, size);
cudaMalloc((void **) &d_f, size);
cudaMalloc((void **) &d_g, size);
cudaMalloc((void **) &d_p, size);
cudaMalloc((void **) &d_rhs, 2 * size);

initialize<<<dim3((NX + TPB - 1) / TPB, (NY + TPB - 1) / TPB), dim3(TPB, TPB)>>>(d_u, d_v, d_p);

cudaMemcpy(h_u, d_u, size, cudaMemcpyDeviceToHost);
cudaMemcpy(h_v, d_v, size, cudaMemcpyDeviceToHost);
cudaMemcpy(h_p, d_p, size, cudaMemcpyDeviceToHost);

for (int step = 0; step < NUM_STEPS; step++) {
    compute_fg<<<dim3((NX + TPB - 1) / TPB, (NY + TPB - 1) / TPB), dim3(TPB, TPB)>>>(d_u, d_v, d_f, d_g);
    cudaMemcpy(h_f, d_f, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_g, d_g, size, cudaMemcpyDeviceToHost);

    compute_rhs<<<dim3((NX + TPB - 1) / TPB, (NY + TPB - 1) / TPB), dim3(TPB, TPB)>>>(d_u, d_v, d_f, d_g, d_rhs);
    cudaMemcpy(h_rhs, d_rhs, 2 * size, cudaMemcpyDeviceToHost);

    for (int itr = 0; itr < MAX_ITERS; itr++) {
        compute_p<<<dim3((NX + TPB - 1) / TPB, (NY + TPB - 1) / TPB), dim3(TPB, TPB)>>>(d_p, d_rhs);
        }
    update_u<<<dim3((NX + TPB - 1) / TPB, (NY + TPB - 1) / TPB), dim3(TPB, TPB)>>>(d_u, d_f, d_p);
    update_v<<<dim3((NX + TPB - 1) / TPB, (NY + TPB - 1) / TPB), dim3(TPB, TPB)>>>(d_v, d_g, d_p);
    }
cudaMemcpy(h_u, d_u, size, cudaMemcpyDeviceToHost);
cudaMemcpy(h_v, d_v, size, cudaMemcpyDeviceToHost);
cudaMemcpy(h_p, d_p, size, cudaMemcpyDeviceToHost);

for (int i = 0; i < NX; i++) {
for (int j = 0; j < NY; j++) {
printf("u[%d][%d] = %lf, v[%d][%d] = %lf, p[%d][%d] = %lf\n", i, j, h_u[i * NY + j], i, j, h_v[i * NY + j], i, j, h_p[i * NY + j]);
        }
    }

free(h_u);
free(h_v);
free(h_f);
free(h_g);
free(h_p);
free(h_rhs);

cudaFree(d_u);
cudaFree(d_v);
cudaFree(d_f);
cudaFree(d_g);
cudaFree(d_p);
cudaFree(d_rhs);

return 0;
}

