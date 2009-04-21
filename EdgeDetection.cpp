#include "EdgeDetection.h"
#include "Convolution.h"
#include "ImageHandler.h"
#include "Timer.h"

#include <iostream>
#include <cmath>
#include <limits>

using namespace std;

#define USE_GPU

EdgeDetection::EdgeDetection()
:
	xGradient(NULL),
	yGradient(NULL)
{
	initializeDevice();
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
	// Perform smoothing
	startTimer();
	smoothImage();
	this->imgHandler.writeImage(this->imgHandler.getMatrix(), "smoothed.png", true);
	return;
	// Compute image gradient
	float* gradientMagnitude = new float[this->imgHandler.getImagePixelCount()];
	computeImageGradient(gradientMagnitude);
	//this->imgHandler.writeImage(gradientMagnitude, "gradient.png", false);

	// Compute edge directions
	float* edgeDirections = new float[this->imgHandler.getImagePixelCount()];
	computeEdgeDirections(edgeDirections);

	// Snap and classify edge directions
	unsigned* edgeDirectionClassifications = new unsigned[this->imgHandler.getImagePixelCount()];
	classifyEdgeDirections(edgeDirections, edgeDirectionClassifications);
	//this->imgHandler.writeEdgeDirectionImage(edgeDirectionClassifications, gradientMagnitude, "edgeDirections.png");

	// Perform nonmaximum suppression
	suppressNonmaximums(gradientMagnitude, edgeDirectionClassifications);
	//this->imgHandler.writeImage(gradientMagnitude, "gradient.png", true);

	// Perform hysteresis
	float* outputEdges = new float[this->imgHandler.getImagePixelCount()];
	memset(outputEdges, 0, sizeof(float) * this->imgHandler.getImagePixelCount());
	performHysteresis(gradientMagnitude, 3500.0, 50.0, outputEdges);
	stopTimer();
	cout << "Total time was: " << getElapsedTime();

	this->imgHandler.writeImage(outputEdges, "edges.png", false);

	delete[] gradientMagnitude;
	delete[] edgeDirections;
	delete[] edgeDirectionClassifications;
}

void EdgeDetection::exportEdgeImage(const char* filename) const
{

}

void EdgeDetection::computeImageGradient(float* outputImageGradient)
{ 
	this->xGradient = new float[this->imgHandler.getImagePixelCount()];
	this->yGradient = new float[this->imgHandler.getImagePixelCount()];

#ifdef USE_GPU
	performConvolutionGpu(this->imgHandler.getMatrix(), this->imgHandler.getImageWidth(), xGradient, X_GRADIENT);
	performConvolutionGpu(this->imgHandler.getMatrix(), this->imgHandler.getImageWidth(), yGradient, Y_GRADIENT);
#else
	performConvolution(this->imgHandler.getMatrix(), this->imgHandler.getImageWidth(), this->xGradientMask, 3, 4, xGradient);
	performConvolution(this->imgHandler.getMatrix(), this->imgHandler.getImageWidth(), this->yGradientMask, 3, 4, yGradient);
#endif

	for(unsigned i = 0; i < this->imgHandler.getImageWidth(); ++i)
	{
		for(unsigned j = 0; j < this->imgHandler.getImageWidth(); ++j)
		{
			unsigned matrixIndex = i * this->imgHandler.getImageWidth() + j;
			outputImageGradient[matrixIndex] = fabs(xGradient[matrixIndex]) + fabs(yGradient[matrixIndex]);
		}
	}
}

void EdgeDetection::computeEdgeDirections(float* outputEdgeDirections) const
{
	for(unsigned i = 0; i < this->imgHandler.getImagePixelCount(); ++i)
	{
		outputEdgeDirections[i] = static_cast<float>(atan2(xGradient[i], yGradient[i]) * (180 / 3.14159265) + 180.0);
	}
}

void EdgeDetection::classifyEdgeDirections(float* edgeDirections, unsigned* edgeDirectionClassifications) const
{
	for(unsigned i = 0; i < this->imgHandler.getImagePixelCount(); ++i)
	{
		float edgeDirection = edgeDirections[i];
		if(
			(edgeDirection >= 0.0 && edgeDirection < 22.5) ||
			(edgeDirection >= 157.5 && edgeDirection < 202.5) || 
			(edgeDirection >= 337.5 && edgeDirection <= 360.0)
		)
		{
			// -
			edgeDirectionClassifications[i] = 0;
		}
		else if((edgeDirection >= 22.5 && edgeDirection < 67.5) || (edgeDirection >= 202.5 && edgeDirection < 247.5))
		{
			// /
			edgeDirectionClassifications[i] = 1;
		}
		else if((edgeDirection >= 67.5 && edgeDirection < 112.5) || (edgeDirection >= 247.5 && edgeDirection < 292.5))
		{
			// |
			edgeDirectionClassifications[i] = 2;
		}
		else if((edgeDirection >= 112.5 && edgeDirection < 157.5) || (edgeDirection >= 292.5 && edgeDirection < 337.5))
		{
			/* \ */
			edgeDirectionClassifications[i] = 3;
		}
		else
		{
			cerr << "Classifying bad edge direction as 0.  Edge direction was " << edgeDirection << endl;
			edgeDirectionClassifications[i] = 0;
		}
	}
}

void EdgeDetection::suppressNonmaximums(float* imageGradient, unsigned* edgeDirectionClassifications) const
{
	for(unsigned i = 0; i < imgHandler.getImageWidth(); ++i)
	{
		for(unsigned j = 0; j < imgHandler.getImageWidth(); ++j)
		{
			unsigned pixelIndex = i * imgHandler.getImageWidth() + j;
			//cout << "Pixel (" << i << ", " << j << ")" << endl;
			//cout << "\tValue: " << imageGradient[pixelIndex] << ", Direction: " << edgeDirectionClassifications[pixelIndex] << "." << endl;
			int clockwisePerpendicularIndex = 
				getClockwisePerpendicularIndex(i, j, edgeDirectionClassifications[pixelIndex]);		
			float clockwisePerpendicularValue;
			if(clockwisePerpendicularIndex == -1)
			{
				clockwisePerpendicularValue = 0;
			}
			else
			{
				clockwisePerpendicularValue = imageGradient[clockwisePerpendicularIndex];
			}
			//cout << clockwisePerpendicularValue << "." << endl;
			int counterClockwisePerpendicularIndex = 
				getCounterClockwisePerpendicularIndex(i, j, edgeDirectionClassifications[pixelIndex]);
			float counterClockwisePerpendicularValue;
			if(counterClockwisePerpendicularIndex == -1)
			{
				counterClockwisePerpendicularValue = 0;
			}
			else
			{
				counterClockwisePerpendicularValue = imageGradient[counterClockwisePerpendicularIndex];
			}
			//cout << counterClockwisePerpendicularValue << "." << endl;
			if
			(
				imageGradient[pixelIndex] <= clockwisePerpendicularValue ||
				imageGradient[pixelIndex] <= counterClockwisePerpendicularValue
			)
			{
				//cout << "\tPixel suppressed." << endl;
				imageGradient[pixelIndex] = 0;
			}
			else
			{
				//cout << "\tPixel retained." << endl;
			}
		}
	}
}

int EdgeDetection::getClockwisePerpendicularIndex(unsigned i, unsigned j, unsigned edgeDirectionClassification) const
{
	int clockwisePerpendicularI;
	int clockwisePerpendicularJ; 
	switch(edgeDirectionClassification)
	{
		case 0:
			clockwisePerpendicularI = i + 1;
			clockwisePerpendicularJ = j;
			break;
		case 1:
			clockwisePerpendicularI = i - 1;
			clockwisePerpendicularJ = j + 1;
			break;
		case 2:
			clockwisePerpendicularI = i;
			clockwisePerpendicularJ = j + 1;
			break;
		case 3:
			clockwisePerpendicularI = i + 1;
			clockwisePerpendicularJ = j + 1;
			break;
	}
	
	//cout << "\tClockwise perpendicular pixel: (" << clockwisePerpendicularI << ", " << clockwisePerpendicularJ << ") = ";

	if
	(
		clockwisePerpendicularI < 0 || clockwisePerpendicularJ < 0 ||
		clockwisePerpendicularI >= static_cast<int>(this->imgHandler.getImageWidth()) || 
		clockwisePerpendicularJ >= static_cast<int>(this->imgHandler.getImageWidth())
	)
	{
		return -1;
	}
	else
	{
		return clockwisePerpendicularI * imgHandler.getImageWidth() + clockwisePerpendicularJ;
	}
}

int EdgeDetection::getCounterClockwisePerpendicularIndex(unsigned i, unsigned j, unsigned edgeDirectionClassification) const
{
	int counterClockwisePerpendicularI;
	int counterClockwisePerpendicularJ; 
	switch(edgeDirectionClassification)
	{
		case 0:
			counterClockwisePerpendicularI = i - 1;
			counterClockwisePerpendicularJ = j;
			break;
		case 1:
			counterClockwisePerpendicularI = i + 1;
			counterClockwisePerpendicularJ = j - 1;
			break;
		case 2:
			counterClockwisePerpendicularI = i;
			counterClockwisePerpendicularJ = j - 1;
			break;
		case 3:
			counterClockwisePerpendicularI = i - 1;
			counterClockwisePerpendicularJ = j - 1;
			break;
	}

	//cout << "\tCounterclockwise perpendicular pixel: (" << counterClockwisePerpendicularI << ", " << counterClockwisePerpendicularJ << ") = ";

	if
	(
		counterClockwisePerpendicularI < 0 || counterClockwisePerpendicularJ < 0 ||
		counterClockwisePerpendicularJ >= static_cast<int>(this->imgHandler.getImageWidth()) || 
		counterClockwisePerpendicularJ >= static_cast<int>(this->imgHandler.getImageWidth())
	)
	{
		return -1;
	}
	else
	{
		return counterClockwisePerpendicularI * imgHandler.getImageWidth() + counterClockwisePerpendicularJ;
	}
}

void EdgeDetection::performHysteresis
(
	float* gradientImage, float highThreshold, float lowThreshold, float* outputEdges
) 
{
	for(int i = 0; i < static_cast<int>(this->imgHandler.getImageWidth()); ++i)
	{
		for(int j = 0; j < static_cast<int>(this->imgHandler.getImageWidth()); ++j)
		{
			unsigned pixelIndex = i * this->imgHandler.getImageWidth() + j;
			// Mark out borders and all pixels below the high threshold.
			if(gradientImage[pixelIndex] > highThreshold)
			{
				this->visitedPixels.insert(pixelIndex);
				outputEdges[pixelIndex] = 1.0;
				visitNeighbors(i, j, lowThreshold, gradientImage, outputEdges);
			}
		}
	}
}

void EdgeDetection::visitNeighbors(int i, int j, float lowThreshold, float* gradientImage, float* outputEdges)
{
	int pixelIndex = i * this->imgHandler.getImageWidth() + j;

	if
	(
		i == 0 || j == 0 || 
		i == this->imgHandler.getImageWidth() - 1 || j == this->imgHandler.getImageWidth() - 1 ||
		this->visitedPixels.find(pixelIndex) != this->visitedPixels.end() ||
		gradientImage[pixelIndex] < lowThreshold
	)
	{
		this->visitedPixels.insert(pixelIndex);
		return;
	}

	outputEdges[pixelIndex] = 1.0;
	this->visitedPixels.insert(pixelIndex);

	visitNeighbors(i - 1, j - 1, lowThreshold, gradientImage, outputEdges);
	visitNeighbors(i - 1, j, lowThreshold, gradientImage, outputEdges);
	visitNeighbors(i - 1, j + 1, lowThreshold, gradientImage, outputEdges);
	visitNeighbors(i, j + 1, lowThreshold, gradientImage, outputEdges);
	visitNeighbors(i + 1, j + 1, lowThreshold, gradientImage, outputEdges);
	visitNeighbors(i + 1, j, lowThreshold, gradientImage, outputEdges);
	visitNeighbors(i + 1, j - 1, lowThreshold, gradientImage, outputEdges);
	visitNeighbors(i, j - 1, lowThreshold, gradientImage, outputEdges);
}

void EdgeDetection::smoothImage()
{
	float* outputMatrix = new float[this->imgHandler.getImagePixelCount()];
#ifdef USE_GPU
	performConvolutionGpu
	(
		this->imgHandler.getMatrix(),
		this->imgHandler.getImageWidth(),
		outputMatrix,
		GAUSSIAN
	);
#else
	performConvolution
	(
		this->imgHandler.getMatrix(), 
		this->imgHandler.getImageWidth(), 
		gaussianMask, 
		5, 
		gaussianMaskWeight,
		outputMatrix
	);
#endif
	this->imgHandler.setMatrix(outputMatrix);
}

void EdgeDetection::performConvolution(const float* inputMatrix, int matrixWidth, const float* mask, int maskWidth, float maskWeight, float* outputMatrix) const
{
	int maskOffset = (maskWidth - 1) / 2;

	for(int outputRow = 0; outputRow < matrixWidth; ++outputRow)
	{
		for(int outputColumn = 0; outputColumn < matrixWidth; ++outputColumn)
		{
			float accumulator = 0;
			for(int maskRow = -maskOffset; maskRow <= maskOffset; ++maskRow)
			{
				for(int maskColumn = -maskOffset; maskColumn <= maskOffset; ++maskColumn)
				{
					int maskIndex = (maskOffset + maskRow) * maskWidth + (maskOffset + maskColumn);
					int matrixRow = outputRow - (maskOffset - 2 - maskRow);
					int matrixColumn = outputColumn - (maskOffset - 2 - maskColumn);
					int matrixIndex = matrixRow * matrixWidth + matrixColumn;

					if(matrixRow >= 0 && matrixColumn >= 0 && matrixRow < matrixWidth && matrixColumn < matrixWidth)
					{
						//cout << "\tMatrix[" << matrixRow << ", " << matrixColumn <<
						//	"](index: " << matrixIndex << "); Mask[" << maskRow << ", " << maskColumn << "]; Output[" <<
						//	outputRow << ", " << outputColumn << "]" << endl;
						//cout << "\t" << mask[maskIndex] << "*" << inputMatrix[matrixIndex] << "=" << mask[maskIndex] * inputMatrix[matrixIndex] << endl;
						accumulator += mask[maskIndex] * inputMatrix[matrixIndex];
					}
				}
			}
			//cout << "Input: " << inputMatrix[outputRow * matrixWidth + outputColumn] <<
			//	"(index: " << outputRow * matrixWidth + outputColumn <<
			//	"), Output: " << accumulator / gaussianMaskWeight << endl;
			outputMatrix[outputRow * matrixWidth + outputColumn] = accumulator / maskWeight;
		}
	}
}

const float EdgeDetection::xGradientMask[9] = 
{
	-1, 0, 1,
	-2, 0, 2,
	-1, 0, 1
};

const float EdgeDetection::yGradientMask[9] = 
{
	 1,  2,  1,
	 0,  0,  0,
	-1, -2, -1
};

const float EdgeDetection::gaussianMask[25] =
{
	2,  4,  5,  4, 2,
	4,  9, 12,  9, 4,
	5, 12, 15, 12, 5,
	4,  9, 12,  9, 4,
	2,  4,  5,  4, 2
};

const float EdgeDetection::gaussianMaskWeight = 159;
void startIt();
int main(char** argv, int argc)
{
	/*if(argc != 3)
	{
		cerr << "Usage: " << "EdgeDetection <input filename> <output filename>" << endl;
		return -1;
	}*/

	EdgeDetection edgeDetector;

	edgeDetector.loadInputImage(/*argv[0]*/"C:\\Documents and Settings\\awade\\Desktop\\NVIDIA CUDA SDK\\projects\\EdgeDetection\\test.png");
	edgeDetector.performEdgeDetection();
	edgeDetector.exportEdgeImage(/*argv[1]*/"");

	cout << "Press any key to exit..." << endl;
	getchar();

	return 0;
}
