#pragma once

#include <string>
#include <memory>

using namespace std;

typedef unsigned short uint16_t;

class ImageHandler
{
	public:
		ImageHandler();
		~ImageHandler();

		void loadImage(const string& filename);
		uint16_t* getMatrix() const;
		void writeImage(uint16_t* imageMatrix, unsigned imageWidth, const string& filename) const;
		unsigned getImagePixelCount() const;
		unsigned getImageWidth() const;

	private:
		uint16_t* matrix;
		unsigned imageWidth;
};