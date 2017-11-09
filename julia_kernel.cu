//Kernel-function for running Julia-set on GPU unsing CUDA

#include <stdio.h>
#include "julia_CUDA.cuh"

__global__
void calculate(complex_t julia_C, int xSize, int ySize, double x_start, double ylower, double step, int *buffer) {
	int n = xSize * ySize; //BUFFERSIZE

	//who am I? And wher do I have to calculate?
	int blocks	= gridDim.x;	//Number of Blocks in the Grid
	int threads	= blockDim.x;	//Number of Threads in Block	
	int blockID	= blockIdx.x;	//Index of current Thread-Block
	int threadID	= threadIdx.x;	//Thread-Id in Block

	//grid-stride loop:
	int index = blockID * threads + threadID;
 	int stride = blocks * threads;
	for (int i = index; i < n; i += stride) {
		/* Calculate the number of iterations until divergence for each pixel.
		If divergence never happens, return MAXITER */
		complex_t c;
		complex_t z;
		complex_t z_old;
		int iter = 0;
		
		// find our starting complex number c -> different! #define PIXEL(i,j) ((i)+(j)*XSIZE)
		c.real = (x_start + step*(i % xSize));
		c.imag = (ylower + step*(i / xSize));
		
		// our starting z is c
		z = c;

		// iterate until we escape
		while(z.real*z.real + z.imag*z.imag < 4) {
	       		// Each pixel in a julia set is calculated using z_n = (z_n-1)² + C
			z_old = z;

			//square z_old
			z.real = (z_old.real * z_old.real) - (z_old.imag * z_old.imag);
  			z.imag = 2 * z_old.real * z_old.imag;
  
			//add c
			z.real = z.real + c.real;
  			z.imag = z.imag + c.imag;
			
			if(++iter==MAXITER) break;		
		}
		buffer[i] = iter;
	}
}
