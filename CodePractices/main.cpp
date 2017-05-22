
#include "ColorPencilDrawing.h"
#include <iostream>

using namespace std;
using namespace cv;

int main()
{
	Mat img = imread("F:\\image\\10.png", IMREAD_GRAYSCALE);

	Mat stroke;
	test(img, stroke);

	imshow("stroke", stroke);
	waitKey();

	return 0;
}