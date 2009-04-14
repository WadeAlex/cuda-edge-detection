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
	unsigned* gradientMagnitude = new unsigned[this->imgHandler.getImagePixelCount()];
	computeImageGradient(this->imgHandler.getMatrix(), gradientMagnitude);
	delete[] gradientMagnitude;
}

void EdgeDetection::exportEdgeImage(const char* filename) const
{

}

void EdgeDetection::computeImageGradient(uint16_t* imageMatrix, unsigned* imageGradient)
{ 
	// To be done on both CPU and GPU.
	// int* xGradient = convolution(imgHandler->getMatrix(), xGradientMask);
	// int* yGradient = convolution(imgHandler->getMatrix(), yGradientMask);
	
}

const int EdgeDetection::xGradientMask[9] = 
{
	-1, 0, 1,
	-2, 0, 2,
	-1, 0, 1
};

const int EdgeDetection::yGradientMask[9] = 
{
	1, 2, 1,
	0, 0, 0,
	-1, -2, -1
};

int main(char** argv, int argc)
{
	/*if(argc != 3)
	{
		cerr << "Usage: " << "EdgeDetection <input filename> <output filename>" << endl;
		return -1;
	}*/

	EdgeDetection edgeDetector;

	edgeDetector.loadInputImage(/*argv[0]*/"c:\\development\\CDA6938\\project2\\test.png");
	edgeDetector.performEdgeDetection();
	edgeDetector.exportEdgeImage(/*argv[1]*/"");

	cout << "Press any key to exit..." << endl;
	getchar();

	return 0;
}