#pragma once

#include <opencv2\opencv.hpp>

// ʹ��Sobel����ͼ����ݶ�
void gradBySobel(const cv::Mat &src, cv::Mat &grad);

// ʹ�ò�ּ���ͼ����ݶ�
void gradByDifference(const cv::Mat &src, cv::Mat &grad);

// �����˶�ģ����
void genMotionBlurKernel(cv::Mat &psf, double len, double angle);

// ����ĳһ������ľ����
void genDirectionKernle(cv::Mat &kernel,double len, double angle);

void genStroke(cv::Mat &grad, cv::Mat &stroke);

void test(const cv::Mat &src, cv::Mat &dst);