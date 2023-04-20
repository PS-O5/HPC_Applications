#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

float *u, *u_new;
float kx = 1.0, ky = 1.0, kz = 1.0;
float dt = 0.01;
int nx = N, ny = N, nz = N;
int n = nx*ny*nz;
int num_threads = 4; // number of threads to use

pthread_barrier_t barrier;

void* set_boundary_conditions(void* args) {
    int thread_id = *((int*) args);
    int i_start = (nx/num_threads) * thread_id;
    int i_end = (nx/num_threads) * (thread_id+1);
    if (thread_id == num_threads-1) i_end = nx; // last thread handles remainder

    for (int i=i_start; i<i_end; i++) {
        for (int j=0; j<ny; j++) {
            for (int k=0; k<nz; k++) {
                if (i==0 || i==nx-1 || j==0 || j==ny-1 || k==0 || k==nz-1) {
                    u[i*ny*nz + j*nz + k] = 1.0;
                    u_new[i*ny*nz + j*nz + k] = 1.0;
                }
            }
        }
    }

    pthread_barrier_wait(&barrier);
    return NULL;
}

int main() {
    u = (float*) malloc(n*sizeof(float));
    u_new = (float*) malloc(n*sizeof(float));
    for (int i=0; i<n; i++) {
        u[i] = 0.0;
        u_new[i] = 0.0;
    }

    pthread_t threads[num_threads];
    int thread_ids[num_threads];
    pthread_barrier_init(&barrier, NULL, num_threads);

    for (int i=0; i<num_threads; i++) {
        thread_ids[i] = i;
        pthread_create(&threads[i], NULL, set_boundary_conditions, (void*) &thread_ids[i]);
    }

    for (int i=0; i<num_threads; i++) {
        pthread_join(threads[i], NULL);
    }

    // rest of code ...
}


/*In this example, I created 'num_threads' threads, and each thread handles a subset of the i values in the outer loop.
Using a 'pthread_barrier' it is ensured that all threads have finished setting the boundary conditions before proceeding with the rest of the code.
TIP: I tried to merge above code with the orignal one in my HPC/CUDA repository, it requires some Device Specific changes to be made in order to execute.*/
