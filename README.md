# HPC_Applications
These are few HPC implementation codes/snippets which I am sharing with permission to share them. You can use them if you want. :)

Some snippets/code might be absent/faulty due to the reasons of licensing/trade-secret/good will in the favor of the company. Also, few changes might be required to make executable. Sorry for this! 
(ー_ー)!!

## CUDA

### 1D_Stress-Strain_Discretization.cu

This is a simple CUDA interpretation of a FEA problem I studied in my bachelors. The reference can be found [here.](http://rip.eng.hawaii.edu/wp-content/uploads/2020/11/me481-femfeaBoundaryConditionsFailureAnalysis-2020f.pdf)

### 3D_Navier_Stokes.cu

Euler's equation for incompressible, inviscid, isotropic fluid using the Finite Difference Method. I suggest creating a function to read data from a file and then writing the results to other (Hint: #include fstream).
  
### 3D_Conduction_Heat_Transfer.cu

Pretty much self explanatory. I recommend above hint.

### Poissons_Equation.cu

The code solves the Poisson equation using the Jacobi iterative method on a 2D grid. The code assumes a Dirichlet boundary condition, where the potential is specified on the boundary of the domain. The Jacobi method updates each point in the grid using a weighted average of its neighbors, and repeats this process until the solution converges to a specified tolerance or a maximum number of iterations is reached.
  
## OpenCL
  
I am a beginner in OpenCL. This code was ran on Intel-AMD:CPU-GPU combo with some changes in code and parameters defined at the compilation and runtime which I believe you are smart enough to figure out :)
  
## OpenMP

### Fibonacci Series.cpp
  
I guess the name is self explainatory. Several runtime parameters are suggested.
  
### Simple_FFT.cpp
  
FFT based on simple radix algorithm. I suggest use of other highly optimised libraries avaiable if you don't want to hustle xD
  
### Mixed-Radix_FFT.cpp
  
Figure it out on your own!

### Vorticity Solver

The header file(.hpp) has class that takes inputs required to solve the vorticity equation for a fluid and solves the equation using input fluid parameters. This class takes in the number of grid points in the x, y, and z directions (`Nx`, `Ny`, and `Nz`), the size of the domain in the x, y, and z directions (`Lx`, `Ly`, and `Lz`), and the kinematic viscosity of the fluid (`nu`) as constructor arguments. It then initializes the fluid variables and allocates memory for them. The `solve` method takes in a time step `dt` and solves the vorticity equation for the given time step using the current values of the fluid variables. It updates the vorticity components `omega_x`, `omega_y`, and `omega_z`, and then uses these updated values to update the velocity components u, v, and w. The update is done using finite differences and explicit time-stepping.
  
## pthreads
  
### 3D_Conduction_Heat_Transfer_HOST_ONLY.cpp

CPU optimised CUDA code for 3D Heat Conduction.
