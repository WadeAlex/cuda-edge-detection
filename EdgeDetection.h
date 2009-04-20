#pragma once

#include "ImageHandler.h"

#include <hash_set>

class EdgeDetection
{
	public:
		EdgeDetection();
		~EdgeDetection();

		void loadInputImage(const char* filename);
		void performEdgeDetection();
		void exportEdgeImage(const char* filename) const;

		static const float xGradientMask[9];
		static const float yGradientMask[9];
		static const float gaussianMask[25];
		static const float gaussianMaskWeight;

	private:
		void smoothImage();
		void computeImageGradient(float* outputImageGradient);
		void computeEdgeDirections(float* outputEdgeDirections) const;
		void classifyEdgeDirections(float* edgeDirections, unsigned* edgeDirectionClassifications) const;
		void suppressNonmaximums(float* imageGradient, unsigned* edgeDirectionClassifications) const;
		void performConvolution(const float* inputMatrix, int matrixWidth, const float* mask, int maskWidth, float maskWeight, float* outputMatrix) const;
		void performHysteresis(float* gradientImage, float highThreshold, float lowThreshold, float* outputEdges);
		int getCounterClockwisePerpendicularIndex(unsigned i, unsigned j, unsigned edgeDirectionClassification) const;
		int getClockwisePerpendicularIndex(unsigned i, unsigned j, unsigned edgeDirectionClassification) const;
		void visitNeighbors(int i, int j, float lowThreshold, float* gradientImage, float* outputEdges);

		ImageHandler imgHandler;
		float* xGradient;
		float* yGradient;
		stdext::hash_set<unsigned> visitedPixels;
};