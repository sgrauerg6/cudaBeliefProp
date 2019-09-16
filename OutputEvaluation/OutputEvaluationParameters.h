/*
 * OutputEvaluationParameters.h
 *
 *  Created on: Sep 13, 2019
 *      Author: scott
 */

#ifndef OUTPUTEVALUATIONPARAMETERS_H_
#define OUTPUTEVALUATIONPARAMETERS_H_

#include <vector>

const float DEFAULT_X_BORDER_GROUND_TRUTH_DISPARITY = 0.0f;
const float DEFAULT_Y_BORDER_GROUND_TRUTH_DISPARITY = 0.0f;

//define the difference in disparity for it to be considered a "significant difference"
//pixels with difference beyond this are called a "bad pixel" if one of the images is the ground truth
//(when value is SMALL_VAL_BP, then any pixel where disparity values are not the same value are considered "bad pixels")
const float SIG_DIFF_THRESHOLD_STEREO_EVAL_THRESHOLD_1 = 0.001f;
const float SIG_DIFF_THRESHOLD_STEREO_EVAL_THRESHOLD_2 = 2.01f;
const float SIG_DIFF_THRESHOLD_STEREO_EVAL_THRESHOLD_3 = 5.01f;
const float SIG_DIFF_THRESHOLD_STEREO_EVAL_THRESHOLD_4 = 10.01f;

const float MAX_DIFF_CAP = SIG_DIFF_THRESHOLD_STEREO_EVAL_THRESHOLD_4;

//don't evaluate the disparities that are within the "border"
//the x-border is from the left/right sides and the
//y-border are from the top/bottom
//initialize to the border of the default ground truth disparity
const float X_BORDER_SIZE_STEREO_EVAL = DEFAULT_X_BORDER_GROUND_TRUTH_DISPARITY;
const float Y_BORDER_SIZE_STEREO_EVAL = DEFAULT_Y_BORDER_GROUND_TRUTH_DISPARITY;

//define a "cap" of the maximum difference between the corresponding disparities
//(make this infinity if you don't want to cap this value)
const float MAX_ABS_DIFF_BETWEEN_CORR_DISP = 65504.0f;

template <class T>
struct OutputEvaluationParameters {
	std::vector<T> output_diff_thresholds_;
	T max_diff_cap_;
	unsigned int x_border_eval_;
	unsigned int y_border_eval_;

	OutputEvaluationParameters() : output_diff_thresholds_{SIG_DIFF_THRESHOLD_STEREO_EVAL_THRESHOLD_1, SIG_DIFF_THRESHOLD_STEREO_EVAL_THRESHOLD_2,
		SIG_DIFF_THRESHOLD_STEREO_EVAL_THRESHOLD_3, SIG_DIFF_THRESHOLD_STEREO_EVAL_THRESHOLD_4}, max_diff_cap_(MAX_DIFF_CAP),
		x_border_eval_(X_BORDER_SIZE_STEREO_EVAL), y_border_eval_(Y_BORDER_SIZE_STEREO_EVAL)
	{ };
};

#endif /* OUTPUTEVALUATIONPARAMETERS_H_ */
