#pragma once

#include <opencv2\opencv.hpp>

// 使用Sobel计算图像的梯度
void gradBySobel(const cv::Mat &src, cv::Mat &grad);

// 使用差分计算图像的梯度
void gradByDifference(const cv::Mat &src, cv::Mat &grad);

// 生成运动模糊核
void genMotionBlurKernel(cv::Mat &psf, double len, double angle);

// 生成某一个方向的卷积核
void genDirectionKernle(cv::Mat &kernel,double len, double angle);

void genStroke(cv::Mat &grad, cv::Mat &stroke);

void test(const cv::Mat &src, cv::Mat &dst);