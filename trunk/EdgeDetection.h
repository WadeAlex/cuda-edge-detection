#pragma once

#include "ImageHandler.h"

class EdgeDetection
{
	public:
		EdgeDetection();
		~EdgeDetection();

		void loadInputImage(const char* filename);
		void performEdgeDetection();
		void exportEdgeImage(const char* filename) const;

		static const int xGradientMask[9];
		static const int yGradientMask[9];
		static const int gaussianMask[25];
	private:
		void smoothImage(int* outputSmoothedImage) const;
		void computeImageGradient(int* inputImageMatrix, unsigned* outputImageGradient) const;
		void performConvolution(int* inputMatrix, int matrixWidth, const int* mask, int maskWidth, int* outputMatrix) const;
		ImageHandler imgHandler;
};