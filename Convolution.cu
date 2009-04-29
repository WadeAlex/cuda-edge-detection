#include <stdlib.h>
#include <time.h>
#include <stdio.h>

#include <cutil.h>

#include "Convolution.h"

texture<float, 2, cudaReadModeElementType> deviceMatrixTexture;
__device__ __constant__ float deviceXGradientMask[9];
__device__ __constant__ float deviceYGradientMask[9];
__device__ __constant__ float deviceGaussianFilterMask[25];

__global__ void deviceGaussianConvolution(float* output, unsigned matrixWidth)
{
	int outputRow = blockIdx.y * blockDim.y + threadIdx.y;
	int outputColumn = blockIdx.x * blockDim.x + threadIdx.x;
 
	float accumulator = 0.0;

#pragma unroll
	for(int i = -2; i <= 2; ++i)
	{
		unsigned matrixColumn = outputColumn + i;
#pragma unroll
		for(int j = -2; j <= 2; ++j)
		{
			accumulator += deviceGaussianFilterMask[(2 + i)* 3 + (2 + j)] * tex2D(deviceMatrixTexture, matrixColumn, outputRow + j);
		}
	}
	
	output[outputRow * matrixWidth + outputColumn] = accumulator / 159;
}

__global__ void deviceComputeGradient(float* outputGradient, unsigned* outputEdgeDirectionClassifications, unsigned matrixWidth)
{
	int outputRow = blockIdx.y * blockDim.y + threadIdx.y;
	int outputColumn = blockIdx.x * blockDim.x + threadIdx.x;


	// Get gradients
	float xAccumulator = 0.0;
	float yAccumulator = 0.0;

#pragma unroll
	for(int i = -1; i <= 1; ++i)
	{
		unsigned matrixColumn = outputColumn + i;
#pragma unroll
		for(int j = -1; j <= 1; ++j)
		{
			int maskIndex = (1 + i)* 3 + (1 + j);
			//printf("%f\n", tex2D(deviceMatrixTexture, matrixColumn, outputRow + j));
			xAccumulator += deviceXGradientMask[maskIndex] * tex2D(deviceMatrixTexture, matrixColumn, outputRow + j);
			yAccumulator += deviceYGradientMask[maskIndex] * tex2D(deviceMatrixTexture, matrixColumn, outputRow + j);
		}
	}

	int matrixIndex = outputRow * matrixWidth + outputColumn;
	
	// Get gradient magnitude
	outputGradient[matrixIndex] = abs(xAccumulator) + abs(yAccumulator);
	
	// Determine edge direction
	float edgeDirection = atan2(yAccumulator, xAccumulator) * (180 / 3.14159265) + 180.0;

	// Classify edge directions
	if((edgeDirection >= 22.5 && edgeDirection < 67.5) || (edgeDirection >= 202.5 && edgeDirection < 247.5))
	{
		outputEdgeDirectionClassifications[matrixIndex] = 1;
	}
	else if((edgeDirection >= 67.5 && edgeDirection < 112.5) || (edgeDirection >= 247.5 && edgeDirection < 292.5))
	{
		outputEdgeDirectionClassifications[matrixIndex] = 2;
	}
	else if((edgeDirection >= 112.5 && edgeDirection < 157.5) || (edgeDirection >= 292.5 && edgeDirection < 337.5))
	{
		outputEdgeDirectionClassifications[matrixIndex] = 3;
	}
	else
	{
		outputEdgeDirectionClassifications[matrixIndex] = 0;
	}
}

void initializeDevice()
{
	unsigned gradientMaskSize = 9 * sizeof(float);
	unsigned gaussianMaskSize = 25 * sizeof(float);

	// Copy kernels to device.
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(deviceXGradientMask, xGradientMask, gradientMaskSize, 0, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(deviceYGradientMask, yGradientMask, gradientMaskSize, 0, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(deviceGaussianFilterMask, gaussianMask, gaussianMaskSize, 0, cudaMemcpyHostToDevice));
}

void computeGradient(const float* inputMatrix, int matrixWidth, float* outputGradient, unsigned* outputEdgeDirections)
{
	// Create timer.
    //unsigned int timer = 0;
    //CUT_SAFE_CALL(cutCreateTimer(&timer));

	// Compute memory sizes.
	unsigned matrixMemorySize = matrixWidth * matrixWidth * sizeof(float);
	
	// Set up device arrays.
	cudaArray* deviceMatrixArray = NULL;
	float* deviceGradient = NULL;
	unsigned* deviceEdgeDirections = NULL;
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
	cudaMallocArray(&deviceMatrixArray, &channelDesc, matrixWidth, matrixWidth);
	CUDA_SAFE_CALL(cudaMalloc((void**)&deviceGradient, matrixMemorySize));
	CUDA_SAFE_CALL(cudaMalloc((void**)&deviceEdgeDirections, matrixWidth * matrixWidth * sizeof(unsigned)));

	// Copy inputs to device.
	CUDA_SAFE_CALL(cudaMemcpyToArray(deviceMatrixArray, 0, 0, inputMatrix, matrixMemorySize, cudaMemcpyHostToDevice));

	// Set up image matrix as a texture.
	deviceMatrixTexture.addressMode[0] = cudaAddressModeClamp;
	deviceMatrixTexture.addressMode[1] = cudaAddressModeClamp;
	cudaBindTextureToArray(deviceMatrixTexture, deviceMatrixArray);

	// Start timer.
	//CUT_SAFE_CALL(cutStartTimer(timer));

	// Do it!
	dim3 dimGrid(matrixWidth / 16, matrixWidth / 16);
	dim3 dimBlock(16, 16);

	deviceGaussianConvolution<<<dimGrid, dimBlock>>>(deviceGradient, matrixWidth);
	CUDA_SAFE_CALL(cudaMemcpy(outputGradient, deviceGradient, matrixMemorySize, cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaUnbindTexture(deviceMatrixTexture));
	CUDA_SAFE_CALL(cudaMemcpyToArray(deviceMatrixArray, 0, 0, outputGradient, matrixMemorySize, cudaMemcpyHostToDevice));
	cudaBindTextureToArray(deviceMatrixTexture, deviceMatrixArray);
	
	deviceComputeGradient<<<dimGrid, dimBlock>>>(deviceGradient, deviceEdgeDirections, matrixWidth);

	// Check for errors.
	CUT_CHECK_ERROR("Kernel execution failed!");

	// Copy device result to host.
	CUDA_SAFE_CALL(cudaMemcpy(outputGradient, deviceGradient, matrixMemorySize, cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaMemcpy(outputEdgeDirections, deviceEdgeDirections, matrixMemorySize, cudaMemcpyDeviceToHost));

	// Stop and destroy timer, print results.
    //CUT_SAFE_CALL(cutStopTimer(timer));
    //printf("Processing time for %dx%d matrix: %f ms\n", matrixWidth, matrixWidth, cutGetTimerValue(timer));
    //CUT_SAFE_CALL(cutDeleteTimer(timer));

	CUDA_SAFE_CALL(cudaFreeArray(deviceMatrixArray));
	CUDA_SAFE_CALL(cudaFree(deviceGradient));
	CUDA_SAFE_CALL(cudaFree(deviceEdgeDirections));
	CUDA_SAFE_CALL(cudaUnbindTexture(deviceMatrixTexture));
}
