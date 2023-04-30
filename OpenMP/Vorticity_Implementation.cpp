#include "Vorticity_Class.hpp"


int main() {
// Initialize the fluid parameters
const double Lx = 1.0;
const double Ly = 1.0;
const double Lz = 1.0;
const int Nx = 100;
const int Ny = 100;
const int Nz = 100;
const double nu = 0.1;
const double dt = 0.01;
const double t_end = 1.0;
// Initialize the domain decomposition parameters
const int num_x_domains = 2;
const int num_y_domains = 2;
const int num_z_domains = 2;

// Create an instance of the FluidSolver class
FluidSolver solver(Lx, Ly, Lz, Nx, Ny, Nz, nu, dt, num_x_domains, num_y_domains, num_z_domains);

// Solve the fluid equations
solver.solve(t_end);

return 0;
}
