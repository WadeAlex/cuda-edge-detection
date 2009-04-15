#include "EdgeDetection.h"
#include "Convolution.h"
#include "ImageHandler.h"

#include <iostream>

using namespace std;

EdgeDetection::EdgeDetection()
{

}

EdgeDetection::~EdgeDetection()
{

}

void EdgeDetection::loadInputImage(const char* filename)
{
	this->imgHandler.loadImage(filename);
}

void EdgeDetection::performEdgeDetection()
{
	int* smoothedImage = new int[this->imgHandler.getImagePixelCount()];
	smoothImage(smoothedImage);
	unsigned* gradientMagnitude = new unsigned[this->imgHandler.getImagePixelCount()];
	computeImageGradient(smoothedImage, gradientMagnitude);
	
	this->imgHandler.writeImage
	(
		reinterpret_cast<uint16_t*>(gradientMagnitude),
		this->imgHandler.getImageWidth(),
		"test.png"
	);

	delete[] gradientMagnitude;
	delete[] smoothedImage;
}

void EdgeDetection::exportEdgeImage(const char* filename) const
{

}

void EdgeDetection::computeImageGradient(int* inputImageMatrix, unsigned* outputImageGradient) const
{ 
	int* xGradient = new int[this->imgHandler.getImagePixelCount()];
	int* yGradient = new int[this->imgHandler.getImagePixelCount()];

	performConvolution(inputImageMatrix, this->imgHandler.getImageWidth(), this->xGradientMask, 3, xGradient);
	performConvolution(inputImageMatrix, this->imgHandler.getImageWidth(), this->yGradientMask, 3, yGradient);

	for(unsigned i = 0; i < this->imgHandler.getImageWidth(); ++i)
	{
		for(unsigned j = 0; j < this->imgHandler.getImageWidth(); ++j)
		{
			unsigned matrixIndex = i * this->imgHandler.getImageWidth() + j;
			outputImageGradient[matrixIndex] = abs(xGradient[matrixIndex]) + abs(yGradient[matrixIndex]);
		}
	}

	delete[] yGradient;
	delete[] xGradient;	
}

void EdgeDetection::smoothImage(int* outputSmoothedImage) const
{
	performConvolution
	(
		reinterpret_cast<int*>(this->imgHandler.getMatrix()), 
		this->imgHandler.getImageWidth(), 
		gaussianMask, 
		5, 
		outputSmoothedImage
	);
}

void EdgeDetection::performConvolution(int* inputMatrix, int matrixWidth, const int* mask, int maskWidth, int* outputMatrix) const
{
	for(int outputRow = 0; outputRow < matrixWidth; ++outputRow)
	{
		for(int outputColumn = 0; outputColumn < matrixWidth; ++outputColumn)
		{
			unsigned accumulator = 0;
			for(int maskRow = 0; maskRow < maskWidth; ++maskRow)
			{
				for(int maskColumn = 0; maskColumn < maskWidth; ++maskColumn)
				{
					int maskIndex = maskRow * maskWidth + maskColumn;
					int matrixRow = outputRow - (maskWidth - 1 - maskRow);
					int matrixColumn = outputColumn - (maskWidth - 1 - maskColumn);
					int matrixIndex = matrixRow * maskWidth + matrixColumn;
					if(matrixRow >= 0 && matrixColumn >= 0)
					{
						accumulator += mask[maskIndex] * inputMatrix[matrixIndex];
					}
				}
			}
		}
	}
}

const int EdgeDetection::xGradientMask[9] = 
{
	-1, 0, 1,
	-2, 0, 2,
	-1, 0, 1
};

const int EdgeDetection::yGradientMask[9] = 
{
	 1,  2,  1,
	 0,  0,  0,
	-1, -2, -1
};

const int EdgeDetection::gaussianMask[25] =
{
	2,  4,  5,  4, 2,
	4,  9, 12,  9, 4,
	5, 12, 15, 12, 5,
	4,  9, 12,  9, 4,
	2,  4,  5,  4, 2
};

int main(char** argv, int argc)
{
	/*if(argc != 3)
	{
		cerr << "Usage: " << "EdgeDetection <input filename> <output filename>" << endl;
		return -1;
	}*/

	EdgeDetection edgeDetector;

	edgeDetector.loadInputImage(/*argv[0]*/"C:\\Development\\CDA6938\\project2\\NVIDIA CUDA SDK\\projects\\EdgeDetection");
	edgeDetector.performEdgeDetection();
	edgeDetector.exportEdgeImage(/*argv[1]*/"");

	cout << "Press any key to exit..." << endl;
	getchar();

	return 0;
}