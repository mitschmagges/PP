//Kernel-functions for running Julia-set on GPU unsing CUDA

// CUDA Kernel function to add the elements of two arrays on the GPU
__global__
//TODO Change that
void add(int n, float *x, float *y)
{
  for (int i = 0; i < n; i++)
      y[i] = x[i] + y[i];
}
