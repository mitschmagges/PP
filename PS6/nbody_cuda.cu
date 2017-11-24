#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <time.h>

#define dT 0.2f
#define G 0.6f
//#define BLOCK_SIZE 32
//#define BLOCK_SIZE 64
//#define BLOCK_SIZE 128
//#define BLOCK_SIZE 256
#define BLOCK_SIZE 512 

// Global variables
int num_planets;
int num_timesteps;

// Host arrays
float2* velocities;
float4* planets;

// Device arrays 
float2* velocities_d;
float4* planets_d;


// Parse command line arguments
void parse_args(int argc, char** argv){
    if(argc != 2){
        printf("Useage: nbody num_timesteps\n");
        exit(-1);
    }
    
    num_timesteps = strtol(argv[1], 0, 10);
}

// Reads planets from planets.txt
void read_planets(){

    FILE* file = fopen("planets256.txt", "r");
    //FILE* file = fopen("planets1024.txt", "r");
    //FILE* file = fopen("planets4096.txt", "r");
    if(file == NULL){
        printf("'planets.txt' not found. Exiting\n");
        exit(-1);
    }

    char line[200];
    fgets(line, 200, file);
    sscanf(line, "%d", &num_planets);

    planets = (float4*)malloc(sizeof(float4)*num_planets);
    velocities = (float2*)malloc(sizeof(float2)*num_planets);

    for(int p = 0; p < num_planets; p++){
        fgets(line, 200, file);
        sscanf(line, "%f %f %f %f %f",
                &planets[p].x,
                &planets[p].y,
                &velocities[p].x,
                &velocities[p].y,
                &planets[p].z);
    }

    fclose(file);
}

// Writes planets to file
void write_planets(int timestep){
    char name[20];
    int n = sprintf(name, "planets_out.txt");

    FILE* file = fopen(name, "wr+");

    for(int p = 0; p < num_planets; p++){
        fprintf(file, "%f %f %f %f %f\n",
                planets[p].x,
                planets[p].y,
                velocities[p].x,
                velocities[p].y,
                planets[p].z);
    }

    fclose(file);
}

// TODO 7. Calculate the change in velocity for p, caused by the interaction with q
__device__ float2 calculate_velocity_change_planet(float4 p, float4 q){
    float2 r;
    r.x = q.x - p.x;
    r.y = q.y - p.y;
    if(r.x == 0 && r.y == 0){
        float2 v = {0.0f, 0.0f};
        return v;
    }
    float abs_dist = sqrt(r.x*r.x + r.y*r.y);
    float dist_cubed = abs_dist*abs_dist*abs_dist;
    float2 dv;
    dv.x = dT*G*q.z/dist_cubed * r.x;
    dv.y = dT*G*q.z/dist_cubed * r.y;
    return dv;
}

// TODO 5. Calculate the change in velocity for my_planet, caused by the interactions with a block of planets
__device__ float2 calculate_velocity_change_block(float4 my_planet, float4* shared_planets) {
    float2 velocity = {0.0f, 0.0f};
    for(int i = 0; i < blockDim.x; i++) {
        float2 tempv = calculate_velocity_change_planet(my_planet, shared_planets[i]);
        velocity.x += tempv.x;
        velocity.y += tempv.y;
    }
    return velocity;
}

// TODO 4. Update the velocities by calculating the planet interactions ==> DONE!
__global__ void update_velocities(float4* planets, float2* velocities, int num_planets){
    // Step 1: Overall declarations and setup for the the update-velocity function
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    float4 my_planet = planets[thread_id];
    
    // Step 2: How the planets get distributed into groups of BLOCK_SIZE, and
    __shared__ float4 shared_planets[BLOCK_SIZE];

    for(int i = 0; i < num_planets; i+=blockDim.x) {
        shared_planets[threadIdx.x] = planets[i + threadIdx.x];
        __syncthreads();
    
        // Step 3: The call to the parallel routine calculate_velocity_change_block which help update the velocities for each time-step.
        float2 tempv = calculate_velocity_change_block(my_planet, shared_planets);

        velocities[thread_id].x += tempv.x;
        velocities[thread_id].y += tempv.y;
        __syncthreads();
    }
}

// TODO 7. Update the positions of the planets using the new velocities
__global__ void update_positions(float4* planets, float2* velocities, int num_planets){
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    planets[thread_id].x += velocities[thread_id].x * dT;
    planets[thread_id].y += velocities[thread_id].y * dT;
}

int main(int argc, char** argv){

    //set timestamp
    clock_t begin = clock();

    parse_args(argc, argv);
    read_planets();

    // TODO 1. Allocate device memory, and transfer data to device ==> DONE!
    // -> Step 1: allocation of an array for the number of planets on the host using cudaMalloc
    cudaMalloc(&planets_d, sizeof(float4)*num_planets);
    // -> Step 2: allocation of an array for velocities on the host using cudaMalloc
    cudaMalloc(&velocities_d, sizeof(float2)*num_planets);
    // -> Step 3: allocation of an array for the number of planets on the device using cudaMalloc
    cudaMemcpy(planets_d, planets, sizeof(float4)*num_planets, cudaMemcpyHostToDevice);
    // -> Step 4: allocation of an array for velocities on the device using cudaMalloc
    cudaMemcpy(velocities_d, velocities, sizeof(float2)*num_planets, cudaMemcpyHostToDevice);
    
    //calculate first time (copy to device)
    clock_t first = clock();
    double time_spent_to_copy_to_device = (double)(first - begin) / CLOCKS_PER_SEC;
    //print time
    printf("Copy-to-Device-Time: %f\n", time_spent_to_copy_to_device);

    // -> Step 1: Calculating the number of blocks used based on BLOCK_SIZE
    int num_blocks = num_planets/BLOCK_SIZE + ((num_planets%BLOCK_SIZE == 0) ? 0 : 1);


    // Main loop
    for(int t = 0; t < num_timesteps; t++) {
        // TODO 2. Call kernels ==> DONE
        // -> Step 2: Correct parallel calls to update_velocites and update_positions functions
        update_velocities<<<num_blocks, BLOCK_SIZE>>>(planets_d, velocities_d, num_planets);
        update_positions<<<num_blocks, BLOCK_SIZE>>>(planets_d, velocities_d, num_planets);
    }

    //calculate first time (Caclulation)
    clock_t second = clock();
    double time_spent_to_calculate = (double)(second - first) / CLOCKS_PER_SEC;
    //print time
    printf("Calculation-Time: %f\n", time_spent_to_calculate);

    // TODO 3. Transfer data back to host
    // -> Step 1: Transfer the position and velocity arrays back to host
    cudaMemcpy(velocities, velocities_d, sizeof(float2)*num_planets, cudaMemcpyDeviceToHost);
    cudaMemcpy(planets, planets_d, sizeof(float4)*num_planets, cudaMemcpyDeviceToHost); 

    // Output
    write_planets(num_timesteps);

    //free Stuff
    free(velocities);
    free(planets);
    cudaFree(planets_d);

    //calculate end time (copy to host)
    clock_t end = clock();
    double time_spent_to_copy_to_host = (double)(end - second) / CLOCKS_PER_SEC;
    //print time
    printf("Copy-to-Host-Time: %f\n", time_spent_to_copy_to_host);
    
    double time_all = (double)(end - begin) / CLOCKS_PER_SEC;
    printf("All-Time: %f\n", time_all);
}

/*
REPORT #####################################################################################################
	-> New PC on Tulpan (i7 7700k / GTX 1080 ti)
a) 
Serial Version: (1 Thread, 21000 Timesteps)
	- 256	Planets: 24.623289s
	- 1024	Planets: 395.421915s (~6min 35s)
	- 4069	Planets: take to much time

Cuda Version: (BLOCK_SIZE 64)
	- 256	Planets:
		Copy-to-Device-Time:	0.160985s
		Calculation-Time:	1.113486s
		Copy-to-Host-Time:	0.026647s
		-> All-Time:		1.301118s
	- 1024	Planets:
		Copy-to-Device-Time: 	0.155725s
		Calculation-Time:	4.020773s
		Copy-to-Host-Time:	0.101277s
		-> All-Time:		4.277775s
	- 4069	Planets:
		Copy-to-Device-Time:	0.151724s
		Calculation-Time:	16.207968s
		Copy-to-Host-Time:	0.409272s
		-> All-Time:		16.768964s

==> SPEEDUP:
	- 256	18.924716x
	- 1024	92.436352x

b)
Cuda Version: (BLOCK_SIZE 32)
	- 256	Planets:
		Copy-to-Device-Time:	0.140635
		Calculation-Time:	1.073448
		Copy-to-Host-Time:	0.029205
		-> All-Time:		1.243288 <-- FASTEST (256 Planets)
	- 1024	Planets:
		Copy-to-Device-Time:	0.148540
		Calculation-Time: 	4.160538
		Copy-to-Host-Time: 	0.105188
		-> All-Time: 		4.414266
	- 4069	Planets:
		Copy-to-Device-Time: 	0.167685
		Calculation-Time: 	16.654356
		Copy-to-Host-Time: 	0.421628
		-> All-Time: 		17.243669 //slowest (4096)

Cuda Version: (BLOCK_SIZE 256)
	- 256	Planets:
		Copy-to-Device-Time: 	0.153521
		Calculation-Time: 	1.120654
		Copy-to-Host-Time: 	0.026555
		-> All-Time: 		1.300730
	- 1024	Planets:
		Copy-to-Device-Time: 	0.144414
		Calculation-Time: 	4.000408
		Copy-to-Host-Time: 	0.100507
		-> All-Time: 		4.245329 <-- FASTEST (1024 Planets)
	- 4069	Planets:
		Copy-to-Device-Time: 	0.169759
		Calculation-Time: 	15.745258
		Copy-to-Host-Time: 	0.397490
		-> All-Time: 		16.312507 <-- FASTEST (4096 Planets)

Cuda Version: (BLOCK_SIZE 512)
	- 256	Planets:
		Copy-to-Device-Time: 	0.162534
		Calculation-Time: 	5.162525
		Copy-to-Host-Time: 	0.143631
		-> All-Time: 		5.468690 //slowest (256)
	- 1024	Planets:
		Copy-to-Device-Time: 	0.141358
		Calculation-Time: 	4.169906
		Copy-to-Host-Time: 	0.104564
		-> All-Time: 		4.415828 //slowest (1024)
	- 4069	Planets:
		Copy-to-Device-Time: 	0.143752
		Calculation-Time: 	16.378144
		Copy-to-Host-Time: 	0.414755
		-> All-Time: 		16.936651

There are some small differences using different BLOCK_SIZEs.
The over all fastest one is the Number 256. (except of run with 256 Planets)
Realy slow is the run with BLCK_SIZE 512 and 256 Planets.
In general the differences are realy small. 
The differences seem to come from a combination of Problem-Size and BLOCK_SIZE.
It also depends on the used hardware. (Which was the same for all the runs)
-> There is also Occupancy in CUDA, which defined as a ratio of active warps per SM to max. warps that can be active at once.
This can help to pick a good BLOCK_SIZE.

*/

