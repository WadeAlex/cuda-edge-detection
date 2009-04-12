#include <stdlib.h>
#include <time.h>
#include <stdio.h>

#include <cutil.h>

// A_DIMENSION should be the actual a dimension + 2 because both sides of the input matrix are padded with zeroes.
#define A_DIMENSION 258
#define H_DIMENSION 7
#define C_DIMENSION A_DIMENSION

texture<float, 2, cudaReadModeElementType> deviceATexture;
__device__ __constant__ float deviceH[H_DIMENSION * H_DIMENSION];

__global__ void unoptimizedConvolution(float* a, float* h, float* c)
{
	int outputRow = threadIdx.y;
	int outputColumn = threadIdx.x;

	float accumulator = 0.0;

	for(int i = 0; i < H_DIMENSION; ++i)
	{
		for(int j = 0; j < H_DIMENSION; ++j)
		{
			if(outputColumn - i >= 0 && outputColumn - i < A_DIMENSION && outputRow - j >= 0 && outputRow - j < A_DIMENSION)
			{
				accumulator += h[i * H_DIMENSION + j] * a[(outputColumn - i) * A_DIMENSION + outputRow - j];
			}
		}
	}

	c[outputRow * A_DIMENSION + outputColumn] = accumulator;
}

__global__ void optimizedConvolution(float* c)
{
	printf("HI\n");
	
	int outputRow = blockIdx.y;
	int outputColumn = blockIdx.x;

	float accumulator = 0.0;

#pragma unroll
	for(unsigned i = 0; i < H_DIMENSION; ++i)
	{
		unsigned aColumn = outputColumn - i;
#pragma unroll
		for(unsigned j = 0; j < H_DIMENSION; ++j)
		{
			accumulator += deviceH[i * H_DIMENSION + j] * tex2D(deviceATexture, aColumn, outputRow - j);
		}
	}

	c[outputRow * C_DIMENSION + outputColumn] = accumulator;
}

float generateRandomValue(float lowValue, float highValue)
{
	return (float)(rand() / ((float)RAND_MAX + 1.0) * (highValue - lowValue) + lowValue);
}

void populateMatrix(float* matrix, unsigned dimension, float initialValue, bool pad)
{
	for(unsigned i = 0; i < dimension; ++i)
	{
		for(unsigned j = 0; j < dimension; ++j)
		{
			if(!pad || (i > 0 && j > 0 && i < dimension - 1 && j < dimension - 1))
			{
				matrix[i * dimension + j] = initialValue;
			}
			else
			{
				matrix[i * dimension + j] = 0;
			}
		}
	}
}

void populateMatrix(float* matrix, unsigned dimension, float lowValue, float highValue, bool pad)
{
	for(unsigned i = 0; i < dimension; ++i)
	{
		for(unsigned j = 0; j < dimension; ++j)
		{
			if(!pad || (i > 0 && j > 0 && i < dimension - 1 && j < dimension - 1))
			{
				matrix[i * dimension + j] = generateRandomValue(lowValue, highValue);
			}
			else
			{
				matrix[i * dimension + j] = 0;
			}
		}
	}
}

void writeToCsvFile(const char* filename, float* matrix, unsigned dimension)
{
	// Open the file
	FILE* outputFile;
	outputFile = fopen(filename, "w");

	// Write values to it
	printf("Writing results to CSV file.\n");
	for(unsigned i = 1; i < dimension - 1; ++i)
	{
		for(unsigned j = 1; j < dimension - 1; ++j)
		{
			fprintf(outputFile, "%f,", matrix[i * dimension + j]);
		}
		fprintf(outputFile, "\n");
	}

	fclose(outputFile);
}

void performConvolution(float* kernel, float* image, float* result)
{
	srand((unsigned)time(NULL));

	// Create timer.
    unsigned int timer = 0;
    CUT_SAFE_CALL(cutCreateTimer(&timer));

	// Compute memory sizes.
	unsigned memSizeA = A_DIMENSION * A_DIMENSION * sizeof(float);
	unsigned memSizeH = H_DIMENSION * H_DIMENSION * sizeof(float);
	unsigned memSizeC = memSizeA;

	// Allocate and initialize host memory.
	float* hostA = (float*)malloc(memSizeA);
	float* hostH = (float*)malloc(memSizeH);
	float* hostC = (float*)calloc(1, memSizeC);
	//populateMatrix(hostA, A_DIMENSION, 0.0, 100.0, true);
	//populateMatrix(hostH, H_DIMENSION, 0.0, 100.0, false);
	populateMatrix(hostA, A_DIMENSION, 7.0, true);
	populateMatrix(hostH, H_DIMENSION, 3.0, false);

	// Set up device arrays.
	cudaArray* deviceAArray = NULL;
	float* deviceCArray = NULL;
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
	cudaMallocArray(&deviceAArray, &channelDesc, A_DIMENSION, A_DIMENSION);
	CUDA_SAFE_CALL(cudaMalloc((void**)&deviceCArray, memSizeC));

	// Copy inputs to device.
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(deviceH, hostH, memSizeH, 0, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpyToArray(deviceAArray, 0, 0, hostA, memSizeA, cudaMemcpyHostToDevice));

	// Set up A and  H as device textures.
	deviceATexture.addressMode[0] = cudaAddressModeClamp;
	deviceATexture.addressMode[1] = cudaAddressModeClamp;
	cudaBindTextureToArray(deviceATexture, deviceAArray);

	// Start timer.
	CUT_SAFE_CALL(cutStartTimer(timer));

	// Do it!
	dim3 dimGrid(C_DIMENSION, C_DIMENSION);
	dim3 dimBlock(16, 16);
	optimizedConvolution<<<dimGrid, dimBlock>>>(deviceCArray);

	// Check for errors.
	CUT_CHECK_ERROR("Kernel execution failed!");

	// Copy device result to host.
	CUDA_SAFE_CALL(cudaMemcpy(hostC, deviceCArray, memSizeC, cudaMemcpyDeviceToHost));

	// Stop and destroy timer, print results.
    CUT_SAFE_CALL(cutStopTimer(timer));
    printf("Processing time for %dx%d matrix: %f ms\n", A_DIMENSION, A_DIMENSION, cutGetTimerValue(timer));
    CUT_SAFE_CALL(cutDeleteTimer(timer));

	// Write result to disk.
	writeToCsvFile("output.csv", hostC, C_DIMENSION);

	// Free memory.
	free(hostA);
	free(hostH);
	free(hostC);

	CUDA_SAFE_CALL(cudaFreeArray(deviceAArray));
	CUDA_SAFE_CALL(cudaFree(deviceCArray));
	CUDA_SAFE_CALL(cudaUnbindTexture(deviceATexture));

	// Wait for user to press key.
	printf("Press a key to continue...\n");
	getc(stdin);
}