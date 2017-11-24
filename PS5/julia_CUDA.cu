#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "julia_CUDA.cuh"
//#include "julia_kernel.cuh"

double x_start=-2.01;
double x_end=1;
double yupper;
double ylower;

double ycenter=1e-6;
double step;

int pixel[XSIZE * YSIZE];

extern void calculate(complex_t julia_C, int xSize, int ySize, double x_start, double ylower, double step, int *buffer);

/*
complex_t square_complex(complex_t c){
  complex_t r;
  r.real = (c.real * c.real) - (c.imag * c.imag);
  r.imag = 2 * c.real * c.imag;
  return r;
}

complex_t add_complex(complex_t a, complex_t b){
  complex_t r;
  r.real = a.real + b.real;
  r.imag = a.imag + b.imag;
  return r;
}

complex_t add_real(complex_t a, int b){
  complex_t r;
  r.real = a.real + b;
  r.imag = a.imag;
  return r;
}
*/

int main(int argc,char **argv) {
	if(argc==1) {
		puts("Usage: JULIA\n");
		puts("Input real and imaginary part. ex: ./julia 0.0 -0.8");
		return 0;
	}

	/* Calculate the range in the y-axis such that we preserve the
	   aspect ratio */
	step = (x_end - x_start) / XSIZE;   
	yupper = ycenter + (step * YSIZE) / 2;
	ylower = ycenter - (step * YSIZE) / 2;

	// Unlike the mandelbrot set where C is the coordinate being iterated, the
	// julia C is the same for all points and can be chosed arbitrarily
	complex_t julia_C;

	// Get the command line args
	julia_C.real = strtod(argv[1], NULL);
	julia_C.imag = strtod(argv[2], NULL);
	
	//alloc space on device
	int *buffer;
	cudaMallocManaged(&buffer, (XSIZE * YSIZE)*sizeof(int));

	calculate<<<1, 256>>>(julia_C, XSIZE, YSIZE, x_start, ylower, step, buffer);
	
	cudaDeviceSynchronize();
	
	//get buffer back
	
	/* create nice image from iteration counts. take care to create it upside down (bmp format) */
	uchar *imgBuffer = malloc(XSIZE*YSIZE*3);
	for(int i=0; i<XSIZE; i++) {
		for(int j=0; j<YSIZE; j++) {
			int p=((YSIZE-j-1)*XSIZE+i)*3;
			fancycolour(imgBuffer+p,pixel[PIXEL(i,j)]);
		}
	}
	/* write image to disk */
	savebmp("julia.bmp", imgBuffer, XSIZE, YSIZE);

	free(imgBuffer);
	cudaFree(buffer);
	return 0;
}
