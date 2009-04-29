#ifndef _EDGE_DETECTION_H_INCLUDED
#define _EDGE_DETECTION_H_INCLUDED

#include "ImageHandler.h"

#include <hash_set>

class EdgeDetection
{
	public:
		EdgeDetection();
		~EdgeDetection();

		void loadInputImage(string filename);
		void performEdgeDetection();

		static const float xGradientMask[9];
		static const float yGradientMask[9];
		static const float gaussianMask[25];
		static const float gaussianMaskWeight;

	private:
		void smoothImage();
		void computeImageGradient();
		void computeEdgeDirections();
		void classifyEdgeDirections() const;
		void suppressNonmaximums() const;
		void performConvolution(const float* inputMatrix, int matrixWidth, const float* mask, int maskWidth, float maskWeight, float* outputMatrix) const;
		void performHysteresis(float* gradientImage, float highThreshold, float lowThreshold, float* outputEdges);
		int getCounterClockwisePerpendicularIndex(unsigned i, unsigned j, unsigned edgeDirectionClassification) const;
		int getClockwisePerpendicularIndex(unsigned i, unsigned j, unsigned edgeDirectionClassification) const;
		void visitNeighbors(int i, int j, float lowThreshold, float* gradientImage, float* outputEdges);

		ImageHandler imgHandler;
		float* xGradient;
		float* yGradient;
		float* gradient;
		float* edgeDirections;
		unsigned* edgeDirectionClassifications;
		stdext::hash_set<unsigned> visitedPixels;
};

#endif