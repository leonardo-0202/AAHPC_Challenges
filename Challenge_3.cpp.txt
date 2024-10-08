%%cuda
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>

#define DIM_SIZE 16384
// Usate un numero minore per testare.

void printmatrix(unsigned char* a)
{
    printf("Matrix: \n");
    for(int i = 0; i < DIM_SIZE; i++)
    {
      for(int j = 0; j < DIM_SIZE; j++)
      {
          printf("%i ", a[i*DIM_SIZE + j]);
      }
      printf("\n");
    }
}

void printvector(unsigned char* a)
{
    printf("Vector: \n");
    for(int j = 0; j < DIM_SIZE; j++)
    {
        printf("%i ", a[j]);
    }
    printf("\n");
}

__global__ void gpu_function(unsigned char *a, unsigned char *b, long int* sum_array){
    for(int row = threadIdx.x + 32 * blockIdx.x; row<DIM_SIZE; row+=7680){
        int temp = 0;
        for(int j = 0; j<DIM_SIZE; j++){
            temp += a[row*DIM_SIZE + j] * b[j]*b[j]*b[j];
        }
        sum_array[row] = temp;
    }
}


long int cpu_function (unsigned char *a, unsigned char *b)
{
    int row , column;
    long int * c;
    c = (long int*) malloc(sizeof(long int)*DIM_SIZE);
    for (row = 0; row < DIM_SIZE; row++)
    {
        long int temp = 0;
        for (column = 0; column < DIM_SIZE; column++)
        {
            temp += a[row*DIM_SIZE+column] * b[column] * b[column] * b[column];
        }
        c[row] = temp;
    }

    long int sum = 0;
    for (int i = 0; i < DIM_SIZE; i++)
    {
        sum += c[i];
    }

    free(c);

    return sum;
}

int main(int argc, char const *argv[])
{
    /// retrive some info about the CUDA device
    int nDevices;
    cudaGetDeviceCount(&nDevices);
    for (int i = 0; i < nDevices; i++)
    {
      cudaDeviceProp prop;
      cudaGetDeviceProperties(&prop, i);
      printf("Device Number: %d\n", i);
      printf("  Device name: %s\n", prop.name);
      printf("  max Blocks Per MultiProcessor: %d\n", prop.maxBlocksPerMultiProcessor);
      printf("  max Threads Per MultiProcessor: %d\n", prop.maxThreadsPerMultiProcessor);
      printf("  max Threads Per Block: %d\n", prop.maxThreadsPerBlock);
      printf("  num SM: %d\n", prop.multiProcessorCount);
      printf("  num bytes sharedMem Per Block: %d\n", prop.sharedMemPerBlock);
      printf("  num bytes sharedMem Per Multiprocessor: %d\n", prop.sharedMemPerMultiprocessor);
      printf("  Memory Clock Rate (KHz): %d\n",
           prop.memoryClockRate);
      printf("  Memory Bus Width (bits): %d\n",
           prop.memoryBusWidth);
      printf("  Peak Memory Bandwidth (GB/s): %f\n\n",
           2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
    }

    srand(time(NULL));

    unsigned char * matr, * vect;
    long int cpu_result;
    long int gpu_result;

    cudaMallocManaged(&vect, sizeof(unsigned char)*DIM_SIZE);
    cudaMallocManaged(&matr, sizeof(unsigned char)*DIM_SIZE*DIM_SIZE);

    clock_t init_begin = clock();

    // initialize matrix A
    for (int i = 0; i < DIM_SIZE; i++)
    {
        for (int j = 0; j < DIM_SIZE; j++)
        {
              matr[i * DIM_SIZE + j] = ((unsigned char)rand()) % 2;
        }
    }

    //printmatrix(matr);

    // initialize vector B
    for (int j = 0; j < DIM_SIZE; j++)
    {
      vect[j] = ((unsigned char)rand()) % 3;
    }

    //printvector(vect);

    clock_t init_end = clock();
    double init_time = ((double)((init_end - init_begin)) * 1000) / CLOCKS_PER_SEC;
    printf("Time elapsed on initialization with size %d = %f ms\n\n", DIM_SIZE, init_time);

    // sequential version of matrix multiplication
    clock_t begin = clock();
    cpu_result = cpu_function(matr, vect);
    clock_t end = clock();
    double time_spent = ((double)((end - begin)) * 1000) / CLOCKS_PER_SEC;
    printf("Time elapsed on naive CPU implementation with size %d = %f ms\n\n", DIM_SIZE, time_spent);


    float  gpu_elapsed_time_ms;

    // some events to count the execution time
    //clock_t st, end;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);

    // Perform the gpu computation
    // Store the result inside gpu_result
    // You have to modified the data allocation!!
    long int* sum_array;
    int deviceId;
    cudaGetDevice(&deviceId);
    cudaMallocManaged(&sum_array, sizeof(long int) * DIM_SIZE);
    
    cudaMemPrefetchAsync(matr, sizeof(unsigned char)*DIM_SIZE*DIM_SIZE, deviceId);
    cudaMemPrefetchAsync(vect, sizeof(unsigned char)*DIM_SIZE, deviceId);
    cudaMemPrefetchAsync(sum_array, sizeof(long)*DIM_SIZE, deviceId);
    gpu_function<<<240, 32>>>(matr, vect, sum_array);

    cudaThreadSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_elapsed_time_ms, start, stop);
    printf("Time elapsed on GPU implementation with size of %d = %f ms.\n\n", DIM_SIZE, gpu_elapsed_time_ms);
    gpu_result = 0;
    for(int i=0; i<DIM_SIZE; i++){
        gpu_result += sum_array[i];
    }
    
    printf(gpu_result == cpu_result ? "OK" : "ERROR");
    printf("\n cpu_result: %d <-> gpu_result: %d", cpu_result, gpu_result);
    cudaFree(matr);
    cudaFree(vect);
    return 0;
}

