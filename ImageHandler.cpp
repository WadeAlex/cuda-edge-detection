#include "ImageHandler.h"

#include <Magick++.h>

#include <iostream>

ImageHandler::ImageHandler()
:
	matrix(NULL),
	imageRowCount(0),
	imageColumnCount(0)
{

}

ImageHandler::~ImageHandler()
{
	if(matrix)
	{
		delete[] matrix;
	}
}

void ImageHandler::loadImage(const string& filename)
{
	try
	{
		Magick::Image img;
		img.read(filename);
		this->imageRowCount = img.rows();
		this->imageColumnCount = img.columns();
		this->matrix = new uint16_t[getImagePixelCount()];
		Magick::PixelPacket* imgPixels = img.getPixels(0, 0, img.columns(), img.rows());
		for(unsigned i = 0; i < img.rows(); ++i)
		{
			for(unsigned j = 0; j < img.columns(); ++j)
			{
				unsigned index = i * img.columns() + j;
				this->matrix[index] = imgPixels[index].red;
			}
		}
	}
	catch (Magick::Exception& e)
	{
		cerr << "Caught exception: " << e.what() << endl;
	}
}

uint16_t* ImageHandler::getMatrix()
{
	return this->matrix;
}

void ImageHandler::writeImage(const string& filename) const
{

}

unsigned ImageHandler::getImagePixelCount() const
{
	return this->imageRowCount * this->imageColumnCount;
}

unsigned ImageHandler::getImageColumnCount() const
{
	return this->imageColumnCount;
}

unsigned ImageHandler::getImageRowCount() const
{
	return this->imageRowCount;
}