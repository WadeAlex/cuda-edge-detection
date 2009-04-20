#pragma once

#include <string>
#include <memory>

using namespace std;

class ImageHandler
{
	public:
		ImageHandler();
		~ImageHandler();

		void loadImage(const string& filename);
		float* getMatrix();
		const float* getMatrix() const;
		void writeImage(float* imageMatrix, const string& filename, bool rescaleImage) const;
		void writeEdgeDirectionImage(unsigned* edgeDirections, float* intensities, const string& filename, float intensityThreshold = 0.0) const;
		unsigned getImagePixelCount() const;
		unsigned getImageWidth() const;
		void setMatrix(float* newMatrix);

	private:
		float* matrix;
		unsigned imageWidth;
		float getMaximumValue(float* matrix) const;
		void rescale(float* matrix) const;
};