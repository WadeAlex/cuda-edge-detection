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
		this->matrix = new float[getImagePixelCount()];
		Magick::PixelPacket* imgPixels = img.getPixels(0, 0, img.columns(), img.rows());
		for(unsigned i = 0; i < img.rows(); ++i)
		{
			for(unsigned j = 0; j < img.columns(); ++j)
			{
				unsigned index = i * img.columns() + j;
				if(i == 0 || j == 0 || i == img.rows() - 1 || j == img.columns() - 1)
				{
					this->matrix[index] = 0;
				}
				else
				{
					this->matrix[index] = static_cast<float>(imgPixels[index].red);
				}
			}
		}
	}
	catch (Magick::Exception& e)
	{
		cerr << "Caught exception: " << e.what() << endl;
	}
}

float* ImageHandler::getMatrix()
{
	return this->matrix;
}

const float* ImageHandler::getMatrix() const
{
	return this->matrix;
}

void ImageHandler::writeImage(float* imageMatrix, const string& filename, bool rescaleImage) const
{
	if(rescaleImage)
	{
		rescale(imageMatrix);
	}

	stringstream imageResolution;
	imageResolution << imageWidth << 'x' << imageWidth;
	
	try
	{
		Magick::Image outputImage(imageResolution.str(), "white");
		//outputImage.type(Magick::GrayscaleType);

		Magick::PixelPacket* outputImagePixels = outputImage.getPixels(0, 0, outputImage.columns(), outputImage.rows());
		for(unsigned i = 0; i < imageWidth; ++i)
		{
			for(unsigned j = 0; j < imageWidth; ++j)
			{
				unsigned pixelIndex = (i * imageWidth + j);
				outputImagePixels[pixelIndex].red = 
					outputImagePixels[pixelIndex].green = 
					outputImagePixels[pixelIndex].blue =
					imageMatrix[pixelIndex] * 65535;
			}
		}

		outputImage.write(filename);
	}
	catch(Magick::Exception& e)
	{
		cerr << "Caught exception: " << e.what() << endl;
	}
}

void ImageHandler::writeEdgeDirectionImage(unsigned* edgeDirections, float* intensities, const string& filename, float intensityThreshold) const
{
	stringstream imageResolution;
	imageResolution << imageWidth << 'x' << imageWidth;

	try
	{
		Magick::Image outputImage(imageResolution.str(), "black");
		//outputImage.type(Magick::GrayscaleType);

		Magick::PixelPacket* outputImagePixels = outputImage.getPixels(0, 0, outputImage.columns(), outputImage.rows());
		for(unsigned i = 0; i < imageWidth; ++i)
		{
			for(unsigned j = 0; j < imageWidth; ++j)
			{
				unsigned pixelIndex = (i * imageWidth + j);
				if(intensities[pixelIndex] >= intensityThreshold)
				{
					switch(edgeDirections[pixelIndex])
					{
						case 0:
							outputImagePixels[pixelIndex].blue = 65535;
							break;
						case 1:
							outputImagePixels[pixelIndex].red = 65535;
							 break;
						case 2:
							outputImagePixels[pixelIndex].green = 65535;
							break;
						case 3:
							outputImagePixels[pixelIndex].red =	outputImagePixels[pixelIndex].green = 65535;
							break;
					}
				}
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

void ImageHandler::setMatrix(float* newMatrix)
{
	delete[] this->matrix;
	this->matrix = newMatrix;
}

float ImageHandler::getMaximumValue(float* matrix) const
{
	float maximumValue = 0;
	for(unsigned i = 0; i < getImagePixelCount(); ++i)
	{
		if(matrix[i] > maximumValue)
		{
			maximumValue = matrix[i];
		}
	}
	return maximumValue;
}

void ImageHandler::rescale(float* matrix) const
{
	float maximumValue = getMaximumValue(matrix);
	for(unsigned i = 0; i < getImagePixelCount(); ++i)
	{
		matrix[i] /= maximumValue;
	}
}