#include "ColorPencilDrawing.h"
#include <vector>

using namespace std;
using namespace cv;

const double PI = 3.1415926;
const double EPS = 1e-12;

void gradBySobel(const cv::Mat & src, cv::Mat & grad)
{
	Mat grad_x, grad_y;
	Mat abs_grad_x, abs_grad_y;

	Sobel(src, grad_x, CV_32F, 1, 0);
	Sobel(src, grad_y, CV_32F, 0, 1);

	addWeighted(grad_x, 0.5, grad_y, 0.5, 0, grad);
}

void gradByDifference(const cv::Mat & src, cv::Mat & grad)
{
	Mat grad_x;
	Mat kernel_x = (Mat_<float>(1, 3) << -1, 0, 1);
	filter2D(src, grad_x, CV_32F, kernel_x);

	Mat grad_y;
	Mat kernel_y = (Mat_<float>(3, 1) << -1, 0, 1);
	filter2D(src, grad_y, CV_32F, kernel_y);
	magnitude(grad_x, grad_y, grad);
}

void genMotionBlurKernel(cv::Mat & psf, double len, double angle)
{
	double half = len / 2;
	double alpha = (angle - floor(angle / 180) * 180) / 180 * PI;
	double cosalpha = cos(alpha);
	double sinalpha = sin(alpha);
	int xsign;
	if (cosalpha < 0)
	{
		xsign = -1;
	}
	else
	{
		if (angle == 90)
		{
			xsign = 0;
		}
		else
		{
			xsign = 1;
		}
	}
	int psfwdt = 1;
	int sx = (int)fabs(half*cosalpha + psfwdt*xsign - len*EPS);
	int sy = (int)fabs(half*sinalpha + psfwdt - len*EPS);
	Mat_<double> psf1(sy, sx, CV_64F);
	Mat_<double> psf2(sy * 2, sx * 2, CV_64F);
	int row = 2 * sy;
	int col = 2 * sx;
	/*为减小运算量，先计算一半大小的PSF*/
	for (int i = 0; i < sy; i++)
	{
		double* pvalue = psf1.ptr<double>(i);
		for (int j = 0; j < sx; j++)
		{
			pvalue[j] = i*fabs(cosalpha) - j*sinalpha;

			double rad = sqrt(i*i + j*j);
			if (rad >= half && fabs(pvalue[j]) <= psfwdt)
			{
				double temp = half - fabs((j + pvalue[j] * sinalpha) / cosalpha);
				pvalue[j] = sqrt(pvalue[j] * pvalue[j] + temp*temp);
			}
			pvalue[j] = psfwdt + EPS - fabs(pvalue[j]);
			if (pvalue[j] < 0)
			{
				pvalue[j] = 0;
			}
		}
	}
	/*将模糊核矩阵扩展至实际大小*/
	for (int i = 0; i < sy; i++)
	{
		double* pvalue1 = psf1.ptr<double>(i);
		double* pvalue2 = psf2.ptr<double>(i);
		for (int j = 0; j < sx; j++)
		{
			pvalue2[j] = pvalue1[j];
		}
	}

	for (int i = 0; i < sy; i++)
	{
		for (int j = 0; j < sx; j++)
		{
			psf2[2 * sy - 1 - i][2 * sx - 1 - j] = psf1[i][j];
			psf2[sy + i][j] = 0;
			psf2[i][sx + j] = 0;
		}
	}
	/*保持图像总能量不变，归一化矩阵*/
	double sum = 0;
	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < col; j++)
		{
			sum += psf2[i][j];
		}
	}
	psf2 = psf2 / sum;
	if (cosalpha > 0)
	{
		flip(psf2, psf2, 0);
	}

	//cout << "psf2=" << psf2 << endl;  
	psf = psf2;
}

void genDirectionKernle(cv::Mat & kernel,double len, double angle)
{
	Mat init = Mat::zeros(Size(2 * len + 1, 2 * len + 1), CV_32F);
	init.row(len) = Scalar_<float>(1.0f); // 水平方向

	Mat mode = getRotationMatrix2D(Point2f(len, len), angle, 1);
	warpAffine(init, kernel, mode, init.size());

	// 对生成的卷积核进行归一化处理
	double sum = 0;
	for (int i = 0; i < kernel.rows; i++)
	{
		float *ptr = kernel.ptr<float>(i);
		for (int j = 0; j < kernel.cols; j++)
			sum += ptr[j];
	}

	kernel /= sum;
}

void genStroke(cv::Mat & grad, cv::Mat & stroke)
{
	int kernel_size = grad.rows / 60;

	vector<Mat> kernel_list(8, Mat());
	vector<Mat> group(8, Mat());

	for (int i = 0; i < 8; i++)
	{
		genDirectionKernle(kernel_list[i], kernel_size, double(i * 180) / 8.0);
		filter2D(grad, group[i], CV_32F, kernel_list[i]);
	}

	// 计算最大方向的响应
	for (int i = 0; i < grad.rows; i++)
	{
		const float* ps = grad.ptr<float>(i);
		for (int j = 0; j < grad.cols; j++)
		{
			int maxValue = 0;
			int index = 0;
			for (int n = 0; n < 8; n++)
			{
				float* pd = group[n].ptr<float>(i);
				if (maxValue < pd[j])
				{
					index = n;
					maxValue = pd[j];
				}
			}
			for (int n = 0; n < 8; n++)
			{
				float* pd = group[n].ptr<float>(i);
				if (n == index)
					pd[j] = ps[j];
				else
					pd[j] = 0;
			}
		}
	}

	// 对最大的响应结果进行卷积
	Mat s = Mat(grad.size(), CV_32F, Scalar(0));
	for (int i = 0; i < 8; i++)
	{
		filter2D(group[i], group[i], CV_32F, kernel_list[i]);

		s += group[i];
	}

	// 将结果映射到[0,1]
	double min, max;
	minMaxLoc(s, &min, &max);

	s = (s - min) / (max - min);

	stroke = 1 - s; // 反转
}

void test(const cv::Mat & src, cv::Mat & dst)
{
	Mat grad;
	gradByDifference(src, grad);

	genStroke(grad, dst);
}
