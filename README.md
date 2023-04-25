# HPC_Applications
These are few HPC implementation codes/snippets which I am sharing with permission to share them. You can use them if you want. :)

Some snippets/code might be absent/faulty due to the reasons of licensing/trade-secret/good will in the favor of the company. Also, few changes might be required to make executable. Sorry for this! 
(ー_ー)!!

## CUDA

### 1D_Stress-Strain_Discretization.cu

This is a simple CUDA interpretation of a FEA problem I studied in my bachelors. The reference can be found [here.](http://rip.eng.hawaii.edu/wp-content/uploads/2020/11/me481-femfeaBoundaryConditionsFailureAnalysis-2020f.pdf)

### 2D_Navier_Stokes.cu

This program generates the flow fields for a 2D Navier-Stokes governed flow. I suggest creating a function to read data from a file and then writing the results to other (Hint: #incclude <fstream>.
  
### 3D_Conduction_Heat_Transfer.cu

Pretty much self explanatory. I recommend above hint.
  
## OpenCL
  
I am a beginner in OpenCL. This code was ran on Intel-AMD:CPU-GPU combo with some changes in code and parameters defined at the compilation and runtime which I believe you are smart enough to figure out :)
  
## OpenMP

### Fibonacci Series.cpp
  
I guess the name is self explainatory. Several runtime parameters are suggested.
  
### Simple_FFT.cpp
  
FFT based on simple radix algorithm. I suggest use of other highly optimised libraries avaiable if you don't want to hustle xD
  
### Mixed-Radix_FFT.cpp
  
Figure it out on your own!
  
##pthreads
  
### 3D_Conduction_Heat_Transfer_HOST_ONLY.cpp

CPU optimised CUDA code for 3D Heat Conduction.  
