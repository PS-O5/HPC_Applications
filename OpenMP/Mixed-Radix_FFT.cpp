#include <iostream>
#include <complex>
#include <vector>
#include <cmath>
#include <omp.h>

using namespace std;

// Compute the mixed-radix FFT recursively
void fft(vector<complex<double>>& x, int n, const vector<int>& radices, const vector<int>& offsets)
{
    if (n == 1) return;

    vector<complex<double>> y(n);

    int r = radices[0];
    int m = n / r;

    // Transpose the data into a r x m matrix
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < r; ++i)
    {
        for (int j = 0; j < m; ++j)
        {
            y[i*m + j] = x[offsets[i] + j*r];
        }
    }

    // Compute the FFT of each column recursively
    #pragma omp parallel for
    for (int i = 0; i < r; ++i)
    {
        fft(y, m, vector<int>(radices.begin() + 1, radices.end()), vector<int>(offsets.begin() + 1, offsets.end()));
    }

    // Transpose the data back into the original vector
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < r; ++i)
    {
        for (int j = 0; j < m; ++j)
        {
            x[offsets[i] + j*r] = y[i + j*r];
        }
    }

    // Compute the radix-r butterflies
    #pragma omp parallel for
    for (int i = 0; i < r/2; ++i)
    {
        complex<double> w = exp(complex<double>(0, 2 * M_PI * i / r));
        for (int j = 0; j < m; ++j)
        {
            complex<double> a = x[offsets[i] + j*r];
            complex<double> b = x[offsets[i + r/2] + j*r] * w;
            x[offsets[i] + j*r] = a + b;
            x[offsets[i + r/2] + j*r] = a - b;
        }
    }
}

// Compute the inverse FFT (iFFT)
void ifft(vector<complex<double>>& x)
{
    // Reverse the order of the input signal
    reverse(x.begin() + 1, x.end());

    // Compute the FFT of the reversed signal
    fft(x, x.size(), {x.size()}, {0});

    // Reverse the order of the output signal and scale it
    reverse(x.begin() + 1, x.end());
    for (auto& a : x)
    {
        a /= x.size();
    }
}

int main()
{
    int n = 12;
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
    x[8] = 0;
    x[9] = 1;
    x[10] = 2;
    x[11] = 3;

    // Define the radices and offsets for the mixed-radix FFT
   
    vector<int> radices = {2, 3, 2};
    vector<int> offsets = {0, 0, 0};

    // Compute the FFT of the input signal
    fft(x, n, radices, offsets);

    // Print the FFT result
    cout << "FFT result:" << endl;
    for (auto a : x)
    {
       cout << a << " ";
    }
    cout << endl;

    // Compute the iFFT of the FFT result
    ifft(x);

    // Print the iFFT result
    cout << "iFFT result:" << endl;
    for (auto a : x)
    {
        cout << a << " ";
    }
    cout << endl;

    return 0;
}
