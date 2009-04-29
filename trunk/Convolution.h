#pragma once

#define IMAGE_WIDTH 256

#define IMAGE_SIZE IMAGE_WIDTH * IMAGE_WIDTH

//#define USE_GPU

const float xGradientMask[9] = 
{
	-1, 0, 1,
	-2, 0, 2,
	-1, 0, 1
};

const float yGradientMask[9] = 
{
	1,  2,  1,
	0,  0,  0,
	-1, -2, -1
};

const float gaussianMask[25] =
{
	2,  4,  5,  4, 2,
	4,  9, 12,  9, 4,
	5, 12, 15, 12, 5,
	4,  9, 12,  9, 4,
	2,  4,  5,  4, 2
};

void initializeDevice();
void computeGradient(const float* inputMatrix, int matrixWidth, float* outputGradient, unsigned* outputEdgeDirectionClassifications);