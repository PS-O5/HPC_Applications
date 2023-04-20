#include <iostream>
#include <cmath>
#include <complex>
#include <vector>
#include <omp.h>

using namespace std;

// Compute the FFT recursively
void fft(vector<complex<double>>& x)
{
    int n = x.size();
    if (n == 1) return;

    vector<complex<double>> even(n/2);
    vector<complex<double>> odd(n/2);

    // Split the signal into even and odd parts
    for (int i = 0; i < n/2; ++i)
    {
        even[i] = x[2*i];
        odd[i] = x[2*i + 1];
    }

    // Compute the FFT of the even and odd parts recursively
    #pragma omp parallel sections
    {
        #pragma omp section
        {
            fft(even);
        }
        #pragma omp section
        {
            fft(odd);
        }
    }

    // Combine the results
    for (int i = 0; i < n/2; ++i)
    {
        complex<double> w = exp(complex<double>(0, -2 * M_PI * i / n)) * odd[i];
        x[i] = even[i] + w;
        x[i + n/2] = even[i] - w;
    }
}

// Compute the inverse FFT
void ifft(vector<complex<double>>& x)
{
    // Take the complex conjugate
    for (int i = 0; i < x.size(); ++i)
    {
        x[i] = conj(x[i]);
    }

    // Compute the FFT
    fft(x);

    // Take the complex conjugate and scale by 1/n
    for (int i = 0; i < x.size(); ++i)
    {
        x[i] = conj(x[i]) / x.size();
    }
}

int main()
{
    int n = 8;
    vector<complex<double>> x(n);

    // Initialize the input signal
    x[0] = 1;
    x[1] = 2;
    x[2] = 3;
    x[3] = 4;
    x[4] = 4;
    x[5] = 3;
    x[6] = 2;
    x[7] = 1;

    // Compute the FFT
    fft(x);

    // Print the result
    cout << "FFT: ";
    for (int i = 0; i < n; ++i)
    {
        cout << x[i] << " ";
    }
    cout << endl;

    // Compute the inverse FFT
    ifft(x);

    // Print the result
    cout << "Inverse FFT: ";
    for (int i = 0; i < n; ++i)
    {
        cout << x[i] << " ";
    }
    cout << endl;

    return 0;
}
