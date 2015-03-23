#include <opencv2/core/core.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/calib3d/calib3d.hpp"
#include <opencv2/imgproc/imgproc.hpp>
#include <stdio.h>
#include "dirent.h"
#include "morpher.h"
#include <regex>
#include <iostream>


using namespace std;
using namespace cv;

vector<Mat> getImagesFromFolder(string folderPath, regex nameFilter)
{
	vector<Mat> images;
	DIR *dir;
	struct dirent *ent;
	if ((dir = opendir(folderPath.c_str())) != nullptr) {
		while ((ent = readdir(dir)) != nullptr) {
			auto fileName(ent->d_name);
			if (regex_match(fileName, nameFilter))
			{
				auto patchToImageFile = folderPath + string(fileName);
				auto image = imread(patchToImageFile);
				images.push_back(image);
				printf("%s\n", patchToImageFile.c_str());
			}
		}
		closedir(dir);
	}
	else {
		throw;
	}
	return images;
}

void showImages(vector<Mat>& images)
{
	auto selectedImage(0);
	imshow("Image", images[selectedImage]);
	for (;;)
	{
		auto keyCode = waitKey();
		if (keyCode == 2424832 && selectedImage > 0) selectedImage--; //Left arrow
		else if (keyCode == 2555904 && selectedImage < images.size() - 1) selectedImage++; //Right arrow
		else if (keyCode == 27) break; //Esc
		else continue;
		imshow("Image", images[selectedImage]);
	}
}

Mat overlay(const Mat& back, const Mat& front, const Mat& backHomography, const Mat& frontHomography, double opacity)
{
	Mat mask(front.size(), CV_8U), b, f, full, backMask(back.size(), CV_8U);
	mask.setTo(Scalar::all(255));
	backMask.setTo(Scalar::all(255));
	warpPerspective(mask, mask, frontHomography, mask.size());
	warpPerspective(front, f, frontHomography, front.size());
	warpPerspective(backMask, backMask, backHomography, back.size());
	warpPerspective(back, b, backHomography, back.size());

	Mat maskInv, i1, i2, i3, result;
	bitwise_not(mask, maskInv);
	bitwise_not(backMask, backMask);

	bitwise_and(b, b, i1, maskInv);
	bitwise_and(f, f, i2, mask);

	bitwise_and(f, f, i3, backMask);
	add(b, i3, full);

	add(i1, i2, result);
	addWeighted(full, 1 - opacity, result, opacity, 0, result);
	return result;
}

vector<Mat> generateIntermediateFrames(const Mat& startImage, const Mat& finishImage, const Mat& offset, int count)
{
	vector<Mat> images;
	Mat e(Mat::eye(3, 3, CV_64F));
	Mat ofinv(offset.inv());
	for (auto i = 0; i < count; ++i)
	{
		Mat h = (e * (1 - i * 1.0 / count)) + offset * (i * 1.0 / count);
		images.push_back(overlay(startImage, finishImage, h, h*ofinv, i * 1.0 / count));
	}
	return images;
}

void showImages(vector<Mat>& images, vector<Mat>& homographies)
{
	Mat e(Mat::eye(3, 3, CV_64F));
	vector<Mat> warpImages;
	for (auto i = 0; i < images.size() - 1; ++i)
	{
		auto intermediateFrames(generateIntermediateFrames(images[i], images[i + 1], homographies[i], 4));
		warpImages.insert(warpImages.end(), intermediateFrames.begin(), intermediateFrames.end());
	}
	warpImages.push_back(Mat());
	warpPerspective(images.back(), warpImages.back(), e, images.back().size());
	/*for (int i = 0; i < warpImages.size(); ++i)
	{
	cout << imwrite("image" + std::to_string(i) + ".jpg", warpImages[i]);
	}*/
	showImages(warpImages);
}

void showOverlayImages(vector<Mat>& images, vector<Mat>& homographies)
{
	vector<Mat> resultImages;
	Mat h(Mat::eye(3, 3, CV_64F));
	for (auto i = 0; i < images.size() - 1; ++i)
	{
		resultImages.push_back(Mat());
		warpPerspective(images[i], resultImages.back(), h.inv(), images.back().size());
		h = homographies[i] * h;
	}
	resultImages.push_back(Mat());
	warpPerspective(images.back(), resultImages.back(), h.inv(), images.back().size());

	/*for (int i = 0; i < resultImages.size(); ++i)
	{
	cout << imwrite("image" + std::to_string(i) + ".jpg", resultImages[i]);
	}
	cout << endl;*/

	showImages(resultImages);
}

Mat overlay(const Mat& back, const Mat& front, const Mat& backHomography, const Mat& frontHomography, const Mat& maskArg)
{
	Mat mask(maskArg), b, f;

	warpPerspective(mask, mask, frontHomography, mask.size());
	warpPerspective(front, f, frontHomography, front.size());

	warpPerspective(back, b, backHomography, back.size());

	Mat result(overlay(b, f, mask));
	return result;
}

vector<Mat> generateMorphIntermediateFrames(const Mat& startImage, const Mat& finishImage, const Mat& offset, int count)
{
	Mat forMorphImage, mask(Mat_<unsigned char>(startImage.size()));
	warpPerspective(startImage, forMorphImage, offset, startImage.size());
	mask = 255;
	warpPerspective(mask, mask, offset, startImage.size());

	vector<Mat> masks;
	auto frames(morpher::dualMorph(forMorphImage, finishImage, mask, count, masks));

	vector<Mat> result;

	Mat e(Mat::eye(3, 3, CV_64F));
	Mat ofinv(offset.inv());
	for (auto i = 0; i < count; ++i)
	{
		Mat h = (e * (1 - i * 1.0 / count)) + offset * (i * 1.0 / count);
		result.push_back(overlay(startImage, frames[i], h, h*ofinv, masks[i]));
	}
	return result;
}

void showMorphImages(vector<Mat>& images, vector<Mat>& homographies)
{
	vector<Mat> warpImages;
	for (auto i = 0; i < images.size() - 1; ++i)
	{
		auto intermediateFrames(generateMorphIntermediateFrames(images[i], images[i + 1], homographies[i], 4));
		warpImages.insert(warpImages.end(), intermediateFrames.begin(), intermediateFrames.end());
	}
	warpImages.push_back(images.back());
	for (int i = 0; i < warpImages.size(); ++i)
	{
	cout << imwrite("image" + std::to_string(i) + ".jpg", warpImages[i]);
	}
	showImages(warpImages);
}

void getKeypoints(const vector<Mat>& images, vector<vector<KeyPoint>>& keypoints, vector<Mat>& descriptors)
{
	keypoints.clear();
	descriptors.clear();

	SiftFeatureDetector detector;
	detector.detect(images, keypoints);

	SiftDescriptorExtractor extractor;
	for (size_t i = 0; i < images.size(); i++)
	{
		descriptors.push_back(Mat());
		extractor.compute(images[i], keypoints[i], descriptors[i]);
		printf("compute %i/%i\n", i + 1, images.size());
	}
}

vector<DMatch> get_good_matches(Mat& descriptor1, Mat& descriptor2, FlannBasedMatcher& matcher)
{
	vector< DMatch > matches;
	matcher.match(descriptor1, descriptor2, matches);
	double max_dist = 0; double min_dist = 100;

	for (auto i = 0; i < descriptor1.rows; i++)
	{
		double dist = matches[i].distance;
		if (dist < min_dist) min_dist = dist;
		if (dist > max_dist) max_dist = dist;
	}

	printf("-- Max dist : %f \n", max_dist);
	printf("-- Min dist : %f \n", min_dist);

	vector< DMatch > good_matches;

	for (auto i = 0; i < descriptor1.rows; i++)
	{
		if (matches[i].distance < 3 * min_dist)
		{
			good_matches.push_back(matches[i]);
		}
	}
	return good_matches;
}

vector<vector<DMatch>> get_good_matches(vector<Mat>& descriptors)
{
	FlannBasedMatcher matcher;
	vector<vector<DMatch>> matches;
	for (auto i = 0; i < descriptors.size() - 1; i++)
	{
		matches.push_back(get_good_matches(descriptors[i], descriptors[i + 1], matcher));
	}
	return matches;
}

Mat getHomography(vector<DMatch>& good_matches, vector<KeyPoint>& keypoints1, vector<KeyPoint>& keypoints2)
{
	vector<Point2f> obj;
	vector<Point2f> scene;

	for (int i = 0; i < good_matches.size(); i++)
	{
		obj.push_back(keypoints1[good_matches[i].queryIdx].pt);
		scene.push_back(keypoints2[good_matches[i].trainIdx].pt);
	}

	return findHomography(obj, scene, CV_RANSAC);
}

vector<Mat> getHomography(vector<vector<DMatch>>& good_matches, vector<vector<KeyPoint>>& keypoints)
{
	vector<Mat> homographies;
	for (auto i = 0; i < good_matches.size(); ++i)
	{
		homographies.push_back(getHomography(good_matches[i], keypoints[i], keypoints[i + 1]));
	}
	return homographies;
}

int main(int argc, char** argv)
{
	auto folderPath("C:\\Users\\Maxim\\Documents\\coursework\\timelapse1\\");
	auto images(getImagesFromFolder(folderPath, regex(".+\\.jpg")));
	if (images.size())
	{
		vector<vector<KeyPoint>> keypoints;
		vector<Mat> descriptors;

		getKeypoints(images, keypoints, descriptors);
		vector<vector<DMatch>> good_mathces(get_good_matches(descriptors));
		auto homographies(getHomography(good_mathces, keypoints));

		//showImages(images, homographies);
		//showOverlayImages(images, homographies);
		showMorphImages(images, homographies);

		return 0;
	}

	return 0;
}