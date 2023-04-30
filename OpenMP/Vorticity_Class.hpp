#include <cmath>
#include <iostream>

class FluidSolver {
public:
    // Constructor
    FluidSolver(int Nx, int Ny, int Nz, float Lx, float Ly, float Lz, float nu) :
        Nx(Nx), Ny(Ny), Nz(Nz), Lx(Lx), Ly(Ly), Lz(Lz), nu(nu)
    {
        // Allocate memory for the fluid variables
        u = new float[Nx * Ny * Nz];
        v = new float[Nx * Ny * Nz];
        w = new float[Nx * Ny * Nz];
        omega_x = new float[Nx * Ny * Nz];
        omega_y = new float[Nx * Ny * Nz];
        omega_z = new float[Nx * Ny * Nz];

        // Calculate the grid spacing
        dx = Lx / (Nx - 1);
        dy = Ly / (Ny - 1);
        dz = Lz / (Nz - 1);

        // Initialize the fluid variables
        for (int k = 0; k < Nz; k++) {
            for (int j = 0; j < Ny; j++) {
                for (int i = 0; i < Nx; i++) {
                    int idx = k * Nx * Ny + j * Nx + i;
                    u[idx] = 0.0;
                    v[idx] = 0.0;
                    w[idx] = 0.0;
                    omega_x[idx] = 0.0;
                    omega_y[idx] = 0.0;
                    omega_z[idx] = 0.0;
                }
            }
        }
    }

    // Destructor
    ~FluidSolver()
    {
        delete[] u;
        delete[] v;
        delete[] w;
        delete[] omega_x;
        delete[] omega_y;
        delete[] omega_z;
    }

void solve(float dt, int num_threads) {
    // Compute the size of each subdomain
    int num_x_domains = num_threads;
    int num_y_domains = num_threads;
    int num_z_domains = num_threads;
    int subdomain_size_x = Nx / num_x_domains;
    int subdomain_size_y = Ny / num_y_domains;
    int subdomain_size_z = Nz / num_z_domains;

    // Update the vorticity in parallel using OpenMP
    #pragma omp parallel for collapse(3) num_threads(num_threads)
    for (int k_domain = 0; k_domain < num_z_domains; k_domain++) {
        for (int j_domain = 0; j_domain < num_y_domains; j_domain++) {
            for (int i_domain = 0; i_domain < num_x_domains; i_domain++) {
                int k_start = k_domain * subdomain_size_z;
                int j_start = j_domain * subdomain_size_y;
                int i_start = i_domain * subdomain_size_x;

                // Compute the upper limits for each subdomain
                int k_end = (k_domain == num_z_domains - 1) ? Nz - 1 : (k_domain + 1) * subdomain_size_z;
                int j_end = (j_domain == num_y_domains - 1) ? Ny - 1 : (j_domain + 1) * subdomain_size_y;
                int i_end = (i_domain == num_x_domains - 1) ? Nx - 1 : (i_domain + 1) * subdomain_size_x;

                // Update the vorticity in the subdomain
                for (int k = k_start; k < k_end; k++) {
                    for (int j = j_start; j < j_end; j++) {
                        for (int i = i_start; i < i_end; i++) {
                            int idx = k * Nx * Ny + j * Nx + i;
                            int idx_right = idx + 1;
                            int idx_left = idx - 1;
                            int idx_top = idx + Nx;
                            int idx_bottom = idx - Nx;
                            int idx_front = idx + Nx * Ny;
                            int idx_back = idx - Nx * Ny;

                            omega_x[idx] += -dt / (2.0 * dy) * (w[idx_top] - w[idx_bottom])
                                            + dt / (2.0 * dz) * (v[idx_front] - v[idx_back])
                                            + nu * dt / (dx * dx) * (omega_x[idx_right] + omega_x[idx_left]
                                                                        - 2.0 * omega_x[idx])
                                            + nu * dt / (dy * dy) * (omega_x[idx_top] + omega_x[idx_bottom]
                                                                        - 2.0 * omega_x[idx])
                                            + nu * dt / (dz * dz) * (omega_x[idx_front] + omega_x[idx_back]
                                                                        - 2.0 * omega_x[idx]);

                            omega_y[idx] += dt / (2.0 * dx) * (w[idx_right] - w[idx_left])
                                            - dt / (2.0 * dz) * (u[idx_front] - u[idx_back])
                                            + nu * dt / (dx * dx) * (omega_y[idx_right] + omega_y[idx_left]
                                                                        - 2.0 * omega_y[idx])
                                            + nu * dt / (dy * dy) * (omega_y[idx_top] + omega_y[idx_bottom] - 2.0 * omega_y[idx])
                                            + nu * dt / (dz * dz) * (omega_y[idx_front] + omega_y[idx_back] - 2.0 * omega_y[idx]);
                                                  omega_z[idx] += -dt / (2.0 * dx) * (v[idx_top] - v[idx_bottom])
                                        + dt / (2.0 * dy) * (u[idx_right] - u[idx_left])
                                        + nu * dt / (dx * dx) * (omega_z[idx_right] + omega_z[idx_left]
                                                                    - 2.0 * omega_z[idx])
                                        + nu * dt / (dy * dy) * (omega_z[idx_top] + omega_z[idx_bottom]
                                                                    - 2.0 * omega_z[idx])
                                        + nu * dt / (dz * dz) * (omega_z[idx_front] + omega_z[idx_back]
                                                                    - 2.0 * omega_z[idx]);
                    }
                }
            }
        }
    }
}

// Update the velocity in parallel using OpenMP
#pragma omp parallel for collapse(3) num_threads(num_threads)
for (int k_domain = 0; k_domain < num_z_domains; k_domain++) {
    for (int j_domain = 0; j_domain < num_y_domains; j_domain++) {
        for (int i_domain = 0; i_domain < num_x_domains; i_domain++) {
            int k_start = k_domain * subdomain_size_z;
            int j_start = j_domain * subdomain_size_y;
            int i_start = i_domain * subdomain_size_x;

            // Compute the upper limits for each subdomain
            int k_end = (k_domain == num_z_domains - 1) ? Nz - 1 : (k_domain + 1) * subdomain_size_z;
            int j_end = (j_domain == num_y_domains - 1) ? Ny - 1 : (j_domain + 1) * subdomain_size_y;
            int i_end = (i_domain == num_x_domains - 1) ? Nx - 1 : (i_domain + 1) * subdomain_size_x;

            // Update the velocity in the subdomain
            for (int k = k_start; k < k_end; k++) {
                for (int j = j_start; j < j_end; j++) {
                    for (int i = i_start; i < i_end; i++) {
                        int idx = k * Nx * Ny + j * Nx + i;
                        u[idx] = u[idx] - dt / (dx) * (omega_z[idx_top] - omega_z[idx_bottom])
                                        + nu * dt / (dx * dx) * (u[idx_right] + u[idx_left]
                                                                    - 2.0 * u[idx])
                                        + nu * dt / (dy * dy) * (u[idx_top] + u[idx_bottom]
                                                                    - 2.0 * u[idx])
                                        + nu * dt / (dz * dz) * (u[idx_front] + u[idx_back]
                                                                    - 2.0 * u[idx]);

                        v[idx] = v[idx] - dt / (dy) * (omega_x[idx_front] - omega_x[idx_back])
                                        + nu * dt / (dx * dx) * (v[idx_right] + v[idx_left]
                                                                    - 2.0 * v[idx])
                                        + nu * dt / (dy * dy) * (v[idx_top] + v[idx_bottom]
                                                                    - 2.0 * v[idx])
                                        + nu * dt / (dz * dz) * (v[idx_front] + v[idx_back]
                                                                    - 2.0 * v[idx]);
                                              w[idx] = w[idx] - dt / (dz) * (omega_y[idx_right] - omega_y[idx_left])
                                        + nu * dt / (dx * dx) * (w[idx_right] + w[idx_left]
                                                                    - 2.0 * w[idx])
                                        + nu * dt / (dy * dy) * (w[idx_top] + w[idx_bottom]
                                                                    - 2.0 * w[idx])
                                        + nu * dt / (dz * dz) * (w[idx_front] + w[idx_back]
                                                                    - 2.0 * w[idx]);
                    }
                }
            }
        }
    }
}

// Delete the temporary arrays
delete[] omega_x;
delete[] omega_y;
delete[] omega_z;

}

private:
// Fluid parameters
int Nx, Ny, Nz;
float Lx, Ly, Lz, nu;
float dx, dy, dz;
// Fluid variables
float *u, *v, *w;      // Velocity components
float *omega_x, *omega_y, *omega_z; // Vorticity components

};
