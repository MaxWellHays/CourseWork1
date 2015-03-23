#include "morpher.h"
#include <opencv2/opencv.hpp>

vector<Mat> morpher::dualMorph(const Mat& startImage, const Mat& finishImage, const Mat& mask, int count, vector<Mat>& producedMasks)
{
	Mat bIm1, bIm2;
	cvtColor(startImage, bIm1, CV_BGR2GRAY);
	cvtColor(finishImage, bIm2, CV_BGR2GRAY);

	Mat flow, flowX, flowY;
	calcOpticalFlowFarneback(bIm1, bIm2, flow, 0.7, 3, 11, 5, 5, 1.1, 0);
	vector<Mat> flowPlanes;
	split(flow, flowPlanes);
	flowX = flowPlanes[0];
	flowY = flowPlanes[1];

	Mat mapX(Mat_<float>(flowX.size())), mapY(Mat_<float>(flowX.size()));

	int w(bIm1.size().width), h(bIm1.size().height);

	for (int y = 0; y < h; y++)
		for (int x = 0; x < w; x++)
			mapX.at<float>(y, x) = x, mapY.at<float>(y, x) = y;

	Mat morph1, morph2, floatMask;
	mask.convertTo(floatMask, CV_32F);
	floatMask /= 255.0;
	flowX = flowX.mul(floatMask);
	flowY = flowY.mul(floatMask);

	Mat whiteIm(Mat_<uchar>(startImage.size(), 255)), producedMask1, producedMask2;

	vector<Mat> result;
	for (int i = 0; i < count; i++)
	{
		float progress = i * 1.0 / count;
		remap(startImage, morph1, mapX - progress * flowX, mapY - progress * flowY, 0);
		remap(whiteIm, producedMask1, mapX - progress * flowX, mapY - progress * flowY, 0);
		remap(finishImage, morph2, mapX + (1 - progress)*flowX, mapY + (1 - progress)*flowY, 0);
		remap(whiteIm, producedMask2, mapX + (1 - progress)*flowX, mapY + (1 - progress)*flowY, 0);

		result.push_back(Mat());
		addWeighted(morph1, 1 - progress, morph2, progress, 0, result.back());
		producedMasks.push_back(Mat());
		addWeighted(producedMask1, 1 - progress, producedMask2, progress, 0, producedMasks.back());
	}
	return result;
}