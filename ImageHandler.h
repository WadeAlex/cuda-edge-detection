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
		uint16_t* getMatrix();
		void writeImage(const string& filename) const;
		unsigned getImagePixelCount() const;
		unsigned getImageRowCount() const;
		unsigned getImageColumnCount() const;

	private:
		uint16_t* matrix;
		unsigned imageRowCount;
		unsigned imageColumnCount;
};