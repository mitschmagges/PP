#ifndef JULIA_H
#define JULIA_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "bitmap.h"

#define XSIZE 2560
#define YSIZE 2048

#define MAXITER 255

void calculate(complex_t julia_C, int xSize, int ySize, double x_start, double ylower, double step, int *buffer);

// note that we are extra careful with preprocessor macros. Adding parenthesises is never the
// wrong choice.
#define PIXEL(i,j) ((i)+(j)*XSIZE)

// In order to work with complex numbers we need a datatype to hold a real and imaginary part
typedef struct complex_t{
	float real;
	float imag;
} complex_t;

// It's probably a good idea to add some functions for working on imaginary numbers
// although this is not necessary
complex_t square_complex(complex_t c);
complex_t add_complex(complex_t a, complex_t b);
complex_t add_real(complex_t a, int b);
complex_t add_imag(complex_t a, int b);

#endif
