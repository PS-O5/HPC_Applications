#include <CL/sycl.hpp>
#include <iostream>

using namespace cl::sycl;

int main() {
  constexpr size_t matrix_size = 1024;

  // Create a queue to submit work to the GPU
  queue q;

  // Allocate memory for the matrices on the host
  std::vector<float> A(matrix_size * matrix_size);
  std::vector<float> B(matrix_size * matrix_size);
  std::vector<float> C(matrix_size * matrix_size);

  // Initialize matrices A and B with random values
  for (size_t i = 0; i < matrix_size * matrix_size; i++) {
    A[i] = static_cast<float>(rand()) / RAND_MAX;
    B[i] = static_cast<float>(rand()) / RAND_MAX;
  }

  // Create buffers to hold the matrices on the device
  buffer<float, 2> buf_A(A.data(), range<2>(matrix_size, matrix_size));
  buffer<float, 2> buf_B(B.data(), range<2>(matrix_size, matrix_size));
  buffer<float, 2> buf_C(C.data(), range<2>(matrix_size, matrix_size));

  // Submit a command group to the GPU
  q.submit([&](handler& h) {
    // Access the buffers in the kernel
    auto a = buf_A.get_access<access::mode::read>(h);
    auto b = buf_B.get_access<access::mode::read>(h);
    auto c = buf_C.get_access<access::mode::write>(h);

    // Define the kernel that performs matrix multiplication
    h.parallel_for(range<2>(matrix_size, matrix_size), [=](id<2> index) {
      float sum = 0.0f;
      for (size_t i = 0; i < matrix_size; i++) {
        sum += a[{index[0], i}] * b[{i, index[1]}];
      }
      c[index] = sum;
    });
  });

  // Wait for the command group to finish
  q.wait();

  // Check the result
  for (size_t i = 0; i < matrix_size; i++) {
    for (size_t j = 0; j < matrix_size; j++) {
      std::cout << C[i * matrix_size + j] << " ";
    }
    std::cout << std::endl;
  }

  return 0;
}
