#include <iostream>
#include <vector>
#include <omp.h>

int main()
{
    int n = 20; // number of terms to compute
    std::vector<int> fib(n);

    #pragma omp parallel
    {
        int id = omp_get_thread_num();
        int num_threads = omp_get_num_threads();

        int chunk_size = n / num_threads;
        int start = id * chunk_size;
        int end = (id == num_threads-1) ? n : (id+1) * chunk_size;

        // Compute Fibonacci series for each thread
        int a = 0, b = 1;
        for (int i = 0; i < start; i++)
        {
            int tmp = b;
            b = a + b;
            a = tmp;
        }

        for (int i = start; i < end; i++)
        {
            fib[i] = b;
            int tmp = b;
            b = a + b;
            a = tmp;
        }
    }

    // Print the Fibonacci series
    std::cout << "Fibonacci series up to " << n << " terms: ";
    for (int i = 0; i < n; i++)
        std::cout << fib[i] << " ";
    std::cout << std::endl;

    return 0;
}
