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
	private:
		void computeImageGradient(uint16_t* imageMatrix, unsigned* imageGradient);
		ImageHandler imgHandler;
};