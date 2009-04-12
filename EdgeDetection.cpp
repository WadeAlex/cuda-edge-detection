#include "EdgeDetection.h"
#include "Convolution.h"

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

}

void EdgeDetection::performEdgeDetection()
{
	performConvolution(0, 0, 0);
}

void EdgeDetection::exportEdgeImage(const char* filename) const
{

}

int main(char** argv, int argc)
{
	/*if(argc != 3)
	{
		cerr << "Usage: " << "EdgeDetection <input filename> <output filename>" << endl;
		return -1;
	}*/

	EdgeDetection edgeDetector;

	edgeDetector.loadInputImage(/*argv[0]*/"");
	edgeDetector.performEdgeDetection();
	edgeDetector.exportEdgeImage(/*argv[1]*/"");

	return 0;
}