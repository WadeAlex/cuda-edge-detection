#include "ImageHandler.h"

#include <Magick++.h>

#include <iostream>
#include <sstream>

ImageHandler::ImageHandler()
:
	matrix(NULL),
	imageWidth(0)
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
		this->imageWidth = img.rows();
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

uint16_t* ImageHandler::getMatrix() const
{
	return this->matrix;
}

void ImageHandler::writeImage(uint16_t* imageMatrix, unsigned imageWidth, const string& filename) const
{
	stringstream imageResolution;
	imageResolution << imageWidth << 'x' << imageWidth;
	
	try
	{
		Magick::Image outputImage(imageResolution.str(), "black");
		outputImage.type(Magick::GrayscaleType);

		Magick::PixelPacket* outputImagePixels = outputImage.getPixels(0, 0, outputImage.columns(), outputImage.rows());
		for(unsigned i = 0; i < imageWidth; ++i)
		{
			for(unsigned j = 0; j < imageWidth; ++j)
			{
				unsigned pixelIndex = (i * imageWidth + j);
				outputImagePixels[pixelIndex].red = 
					outputImagePixels[pixelIndex].green = 
					outputImagePixels[pixelIndex].blue = 
					imageMatrix[pixelIndex];
			}
		}

		outputImage.write(filename);
	}
	catch(Magick::Exception& e)
	{
		cerr << "Caught exception: " << e.what() << endl;
	}
}

unsigned ImageHandler::getImagePixelCount() const
{
	return this->imageWidth * this->imageWidth;
}

unsigned ImageHandler::getImageWidth() const
{
	return this->imageWidth;
}