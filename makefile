NVCC = nvcc

bitmap.o: bitmap.c bitmap.h
	cc -std=c99 -c -o bitmap.o bitmap.c
#	$(NVCC) -c %< -o julia_CUDA

julia.o:
	nvcc -arch=sm_35 -rdc=true -o julia_CUDA julia_CUDA.cu julia_kernel.cu -lcudadevrt
#	$(NVCC) -c %< -o julia_CUDA
		

all: bitmap.o julia.o
	$(NVCC) %^ -o julia_CUDA

