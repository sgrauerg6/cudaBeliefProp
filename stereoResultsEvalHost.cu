/*
Copyright (C) 2009 Scott Grauer-Gray, Chandra Kambhamettu, and Kannappan Palaniappan

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307 USA
*/

//Defines the functions used to evaluate the results of corresponding disparity maps 1 and 2

#include "stereoResultsEvalHostHeader.cuh"


//initialize the stereo results
__host__ void initializeStereoResults(stereoEvaluationResults*& currentStereoEvaluation)
{
	//initialize the total disparity absolute difference and number of corresponding with significant disparity differences to 0
	currentStereoEvaluation->totalDispAbsDiffNoMax = 0.0f;
	currentStereoEvaluation->numSigDiffPixelsThreshold1 = 0;
	currentStereoEvaluation->numSigDiffPixelsThreshold2 = 0;
	currentStereoEvaluation->numSigDiffPixelsThreshold3 = 0;
	currentStereoEvaluation->numSigDiffPixelsThreshold4 = 0;
}

//given the corresponding disparity values currentDispVal1 and currentDispVal2 from disparity maps 1 and 2, update the current stereo evaluation
__host__ void updateStereoEvaluation(float unscaledCurrentDispVal1, float unscaledCurrentDispVal2, stereoEvaluationResults*& currentStereoEvaluation)
{
	//retrieve the absolute difference between corresponding disparity values
	float absDiffBetweenCorrDispVals = abs(unscaledCurrentDispVal2 - unscaledCurrentDispVal1);

	//add the absolute disparity different to the total absolute difference in disparity capped at MAX_ABS_DIFF_BETWEEN_CORR_DISP
	currentStereoEvaluation->totalDispAbsDiffNoMax += min(absDiffBetweenCorrDispVals, MAX_ABS_DIFF_BETWEEN_CORR_DISP);
	currentStereoEvaluation->totalDispAbsDiffWithMax += min(absDiffBetweenCorrDispVals, SIG_DIFF_THRESHOLD_STEREO_EVAL_THRESHOLD_4);

	//if absolute difference is greater than SIG_DIFF_THRESHOLD_STEREO_EVAL, increment the number of corresponding
	//pixels with significantly different disparity
	if (absDiffBetweenCorrDispVals > SIG_DIFF_THRESHOLD_STEREO_EVAL_THRESHOLD_1)
	{
		currentStereoEvaluation->numSigDiffPixelsThreshold1++;
	}
	if (absDiffBetweenCorrDispVals > SIG_DIFF_THRESHOLD_STEREO_EVAL_THRESHOLD_2)
	{
		currentStereoEvaluation->numSigDiffPixelsThreshold2++;
	}
	if (absDiffBetweenCorrDispVals > SIG_DIFF_THRESHOLD_STEREO_EVAL_THRESHOLD_3)
	{
		currentStereoEvaluation->numSigDiffPixelsThreshold3++;
	}
	if (absDiffBetweenCorrDispVals > SIG_DIFF_THRESHOLD_STEREO_EVAL_THRESHOLD_4)
	{
		currentStereoEvaluation->numSigDiffPixelsThreshold4++;
	}
}

//retrieve the "final" stereo evaluation after the evaluation was updated with every set of corresponding disparity values in disparity maps 1 and 2
//(not including the border region)
__host__ void retrieveFinalStereoEvaluation(stereoEvaluationResults*& currentStereoEvaluation, unsigned int widthDisparityMap, unsigned int heightDisparityMap)
{
	//retrieve the size of the disparity map (not including the border)
	unsigned int sizeDispMapWithoutBorder = (heightDisparityMap - 2*Y_BORDER_SIZE_STEREO_EVAL) * (widthDisparityMap - 2*X_BORDER_SIZE_STEREO_EVAL);

	//retrieve the averageDispAbsDiff by dividing the totalDispAbsDiff by the size of the disparity map (not including the border)
	currentStereoEvaluation->averageDispAbsDiffNoMax = ((float)currentStereoEvaluation->totalDispAbsDiffNoMax) / ((float) sizeDispMapWithoutBorder);
	currentStereoEvaluation->averageDispAbsDiffWithMax = ((float)currentStereoEvaluation->totalDispAbsDiffWithMax) / ((float) sizeDispMapWithoutBorder);

	//retrieve the proportion of significantly different disparities by dividing the numSignDiffPixels by the size of the disparity map (not including the border)
	currentStereoEvaluation->propSigDiffPixelsThreshold1 = ((float)currentStereoEvaluation->numSigDiffPixelsThreshold1) / ((float) sizeDispMapWithoutBorder);
	currentStereoEvaluation->propSigDiffPixelsThreshold2 = ((float)currentStereoEvaluation->numSigDiffPixelsThreshold2) / ((float) sizeDispMapWithoutBorder);
	currentStereoEvaluation->propSigDiffPixelsThreshold3 = ((float)currentStereoEvaluation->numSigDiffPixelsThreshold3) / ((float) sizeDispMapWithoutBorder);
	currentStereoEvaluation->propSigDiffPixelsThreshold4 = ((float)currentStereoEvaluation->numSigDiffPixelsThreshold4) / ((float) sizeDispMapWithoutBorder);
}

//retrieve the stereo evaluation results between the unsigned int scaled disparity maps stored in scaledDispMap1Host and scaledDispMap2Host
//(this method is primary for when the input disparity images are being read from a file)
__host__ stereoEvaluationResults* runStereoResultsEvaluationUsedUnsignedIntScaledDispMap(unsigned int* scaledDispMap1Host, unsigned int* scaledDispMap2Host, float scaleFactorDispMap1, float scaleFactorDispMap2, unsigned int widthDisparityMap, unsigned int heightDisparityMap)
{
	//first declare and initialize the stereo evaluation results
	stereoEvaluationResults* stereoResults = new stereoEvaluationResults;
	initializeStereoResults(stereoResults);
	//go through the corresponding disparities in disparity maps 1 and 2 for every non-border pixel and update the stereo results accordingly
	for (unsigned int currentRowInDispMaps = Y_BORDER_SIZE_STEREO_EVAL; currentRowInDispMaps < (heightDisparityMap - Y_BORDER_SIZE_STEREO_EVAL); currentRowInDispMaps++)
	{
		for (unsigned int currentColInDispMaps = X_BORDER_SIZE_STEREO_EVAL; currentColInDispMaps < (widthDisparityMap - X_BORDER_SIZE_STEREO_EVAL); currentColInDispMaps++)
		{
			unsigned int currentIndexPixel = retrieveIndexStereoOrPixelImage(currentColInDispMaps, currentRowInDispMaps, widthDisparityMap, heightDisparityMap);

			//need to divide the intensity pixels in each disparity map by the scale to retrieve the disparity
			updateStereoEvaluation(((float)scaledDispMap1Host[currentIndexPixel]) / ((float)scaleFactorDispMap1), ((float)scaledDispMap2Host[currentIndexPixel]) / ((float)scaleFactorDispMap2), stereoResults);
		}
	}
	//retrieve the final stereo evaluation results including the average absolute difference between corresponding disparities and the proportion
	//of corresponding disparities where the difference is greater than SIG_DIFF_THRESHOLD_STEREO_EVAL
	retrieveFinalStereoEvaluation(stereoResults, widthDisparityMap, heightDisparityMap);
	//return the resulting stereo results
	return stereoResults;
}


//retrieve the stereo evaluation results between the float-valued unscaled disparity maps stored in unscaledDispMap1Host and unscaledDispMap2Host
//(this method is primary for when the disparity data is current stored in main memory)
__host__ stereoEvaluationResults* runStereoResultsEvaluationUseFloatUnscaledDispMap(float* unscaledDispMap1Host, float* unscaledDispMap2Host, unsigned int widthDisparityMap, unsigned int heightDisparityMap)
{
	//first declare and initialize the stereo evaluation results
	stereoEvaluationResults* stereoResults = new stereoEvaluationResults;

	initializeStereoResults(stereoResults);

	//go through the corresponding disparities in disparity maps 1 and 2 for every non-border pixel and update the stereo results accordingly
	for (unsigned int currentRowInDispMaps = Y_BORDER_SIZE_STEREO_EVAL; currentRowInDispMaps < (heightDisparityMap - Y_BORDER_SIZE_STEREO_EVAL); currentRowInDispMaps++)
	{
		for (unsigned int currentColInDispMaps = X_BORDER_SIZE_STEREO_EVAL; currentColInDispMaps < (widthDisparityMap - X_BORDER_SIZE_STEREO_EVAL); currentColInDispMaps++)
		{
			unsigned int currentIndexPixel = retrieveIndexStereoOrPixelImage(currentColInDispMaps, currentRowInDispMaps, widthDisparityMap, heightDisparityMap);

			//disparity maps are unscaled, so need to divide the values in each disparity map by the scale to retrieve the disparity
			updateStereoEvaluation(unscaledDispMap1Host[currentIndexPixel], unscaledDispMap2Host[currentIndexPixel], stereoResults);
		}
	}

	//retrieve the final stereo evaluation results including the average absolute difference between corresponding disparities and the proportion
	//of corresponding disparities where the difference is greater than SIG_DIFF_THRESHOLD_STEREO_EVAL
	retrieveFinalStereoEvaluation(stereoResults, widthDisparityMap, heightDisparityMap);

	//return the resulting stereo results
	return stereoResults;
}

//retrieve the stereo evaluation results between the float-valued unscaled disparity map 1 stored in unscaledDispMap1Host and the unsigned int disparity 
//map stored in scaledDispMap2Host (this method is primary for when the calculated disparity data in unscaledDispMap1Host is current stored in main memory 
//and the "comparison" disparity (such as the ground truth) is stored in scaledDispMap2
__host__ stereoEvaluationResults* runStereoResultsEvaluationUseFloatUnscaledDispMap(float* unscaledDispMap1Host, unsigned int* scaledDispMap2Host, float scaleFactorDispMap2, unsigned int widthDisparityMap, unsigned int heightDisparityMap)
{
	//first declare and initialize the stereo evaluation results
	stereoEvaluationResults* stereoResults = new stereoEvaluationResults;

	initializeStereoResults(stereoResults);

	//go through the corresponding disparities in disparity maps 1 and 2 for every non-border pixel and update the stereo results accordingly
	for (unsigned int currentRowInDispMaps = Y_BORDER_SIZE_STEREO_EVAL; currentRowInDispMaps < (heightDisparityMap - Y_BORDER_SIZE_STEREO_EVAL); currentRowInDispMaps++)
	{
		for (unsigned int currentColInDispMaps = X_BORDER_SIZE_STEREO_EVAL; currentColInDispMaps < (widthDisparityMap - X_BORDER_SIZE_STEREO_EVAL); currentColInDispMaps++)
		{
			unsigned int currentIndexPixel = retrieveIndexStereoOrPixelImage(currentColInDispMaps, currentRowInDispMaps, widthDisparityMap, heightDisparityMap);

			//disparity map 2 is scaledd, so need to divide the values in disparity map 2 by the scale to retrieve the disparity (but not value in disparity map 1)
			updateStereoEvaluation(unscaledDispMap1Host[currentIndexPixel], ((float)scaledDispMap2Host[currentIndexPixel]) / ((float)scaleFactorDispMap2), stereoResults);
		}
	}

	//retrieve the final stereo evaluation results including the average absolute difference between corresponding disparities and the proportion
	//of corresponding disparities where the difference is greater than SIG_DIFF_THRESHOLD_STEREO_EVAL
	retrieveFinalStereoEvaluation(stereoResults, widthDisparityMap, heightDisparityMap);

	//return the resulting stereo results
	return stereoResults;
}

__host__ void printStereoEvaluationResults(stereoEvaluationResults* evaluationResults, FILE* resultsFile)
{
	//fprintf(resultsFile, "Total RMS error: %f \n", evaluationResults->totalDispAbsDiff);
	fprintf(resultsFile, "Average RMS error: %f \n", evaluationResults->averageDispAbsDiffNoMax);
	fprintf(resultsFile, "Average RMS error (with disparity error cap at %f): %f \n", SIG_DIFF_THRESHOLD_STEREO_EVAL_THRESHOLD_4, evaluationResults->averageDispAbsDiffWithMax);
	//fprintf(resultsFile, "Total bad pixels (Threshold 1): %d \n", evaluationResults->numSigDiffPixelsThreshold1);
	fprintf(resultsFile, "Proportion bad pixels (error less than %f): %f \n", SIG_DIFF_THRESHOLD_STEREO_EVAL_THRESHOLD_1, evaluationResults->propSigDiffPixelsThreshold1);
	//fprintf(resultsFile, "Total bad pixels (Threshold 2): %d \n", evaluationResults->numSigDiffPixelsThreshold2);
	fprintf(resultsFile, "Proportion bad pixels (error less than %f): %f \n", SIG_DIFF_THRESHOLD_STEREO_EVAL_THRESHOLD_2, evaluationResults->propSigDiffPixelsThreshold2);
	//fprintf(resultsFile, "Total bad pixels (Threshold 3): %d \n", evaluationResults->numSigDiffPixelsThreshold3);
	fprintf(resultsFile, "Proportion bad pixels (error less than %f): %f \n", SIG_DIFF_THRESHOLD_STEREO_EVAL_THRESHOLD_3, evaluationResults->propSigDiffPixelsThreshold3);
	//fprintf(resultsFile, "Total bad pixels (Threshold 4): %d \n", evaluationResults->numSigDiffPixelsThreshold4);
	fprintf(resultsFile, "Proportion bad pixels (error less than %f): %f \n", SIG_DIFF_THRESHOLD_STEREO_EVAL_THRESHOLD_4, evaluationResults->propSigDiffPixelsThreshold4);
}

__host__ void writeStereoResultsToFile(FILE* currentfp, stereoEvaluationResults* evaluationResults)
{
	fprintf(currentfp, "Total RMS error: %f \n", evaluationResults->totalDispAbsDiffNoMax);
	fprintf(currentfp, "Average RMS error: %f \n", evaluationResults->averageDispAbsDiffNoMax);
	fprintf(currentfp, "Total bad pixels: %d \n", evaluationResults->numSigDiffPixelsThreshold1);
	fprintf(currentfp, "Proportion bad pixels: %f \n", evaluationResults->propSigDiffPixelsThreshold1);
}


