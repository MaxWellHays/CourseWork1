#pragma once
#include <opencv2/core/core.hpp>

using namespace cv;

class morpher
{
public:
	static vector<Mat> dualMorph(const Mat& startImage, const Mat& finishImage, const Mat& mask, int count, vector<Mat>& producedMasks);
};

inline float ofClamp(float value, float min, float max)
{
	return value < min ? min : value > max ? max : value;
}

inline Mat overlay(const Mat& back, const Mat& front, const Mat& mask)
{
	vector<Mat> v;
	split(back, v);
	for (int j = 0; j < back.rows; j++)
	{
		for (int i = 0; i < back.cols; i++)
		{
			float maskValue = pow(mask.at<uchar>(j, i) / 255.0, 2);
			v[0].at<uchar>(j, i) = ofClamp((1.0 - maskValue) * back.at<Vec3b>(j, i)[0] + maskValue*front.at<Vec3b>(j, i)[0], 0, 255);
			v[1].at<uchar>(j, i) = ofClamp((1.0 - maskValue) * back.at<Vec3b>(j, i)[1] + maskValue*front.at<Vec3b>(j, i)[1], 0, 255);
			v[2].at<uchar>(j, i) = ofClamp((1.0 - maskValue) * back.at<Vec3b>(j, i)[2] + maskValue*front.at<Vec3b>(j, i)[2], 0, 255);
		}
	}
	Mat result;
	merge(v, result);
	return result;
}

