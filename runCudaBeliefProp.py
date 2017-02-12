#!/usr/bin/env python
#make executable in bash chmod +x PyRun

import sys
import inspect
import importlib
import os

if __name__ == "__main__":
	refImages = ["\"tsukuba1.pgm\"", "\"conesQuarter2.pgm\"", "\"conesHalf2.pgm\""]
	testImages = ["\"tsukuba2.pgm\"", "\"conesQuarter6.pgm\"", "\"conesHalf6.pgm\""]
	saveDispGpuOutput = ["\"computedDisparityMapTsukubaGPU.pgm\"", "\"computedDisparityConesQuarterGPU.pgm\"", "\"computedDisparityConesHalfGPU.pgm\""]
	saveDispCpuOutput = ["\"computedDisparityMapTsukubaCPU.pgm\"", "\"computedDisparityConesQuarterCPU.pgm\"", "\"computedDisparityConesHalfCPU.pgm\""]
	numPossDispVals = ["16", "63", "90"]
	scaleBp = ["16.0f", "4.0f", "2.0f"]
	groundTruthDisp = ["\"groundTruthDispTsukuba.pgm\"", "\"conesQuarterGroundTruth.pgm\"", "\"conesHalfGroundTruth.pgm\""]
	groundTruthDispScale = ["16.0f", "4.0f", "2.0f"]
	numBpIters = [ "5", "10", "10"]
	numBpLevels = ["5", "6", "7"]
	fileOutput = open("outputPython.txt", "w")
	for i in range(3):
		file = open("bpParametersFromPython.h", "w")
		file.write("#ifndef BP_STEREO_FROM_PYTHON_H\n")
		file.write("#define BP_STEREO_FROM_PYTHON_H\n")
		file.write("#define REF_IMAGE_FROM_PYTHON %s\n" % refImages[i])
		file.write("#define TEST_IMAGE_FROM_PYTHON %s\n" % testImages[i])
		file.write("#define SAVE_DISPARITY_IMAGE_PATH_GPU_FROM_PYTHON %s\n" % saveDispGpuOutput[i])
		file.write("#define SAVE_DISPARITY_IMAGE_PATH_CPU_FROM_PYTHON %s\n" % saveDispCpuOutput[i])
		file.write("#define NUM_POSSIBLE_DISPARITY_VALUES_FROM_PYTHON %s\n" % numPossDispVals[i])
		file.write("#define SCALE_BP_FROM_PYTHON %s\n" % scaleBp[i])
		file.write("#define DEFAULT_GROUND_TRUTH_DISPARITY_FILE_FROM_PYTHON %s\n" % groundTruthDisp[i])
		file.write("#define DEFAULT_GROUND_TRUTH_DISPARITY_SCALE_FROM_PYTHON %s\n" % groundTruthDispScale[i])
		file.write("#define ITER_BP_FROM_PYTHON %s\n" % numBpIters[i])
		file.write("#define LEVELS_BP_FROM_PYTHON %s\n" % numBpLevels[i])
		file.write("#endif")
		file.close()

		os.system("make clean")
		os.system("make")
		os.system("./driverCudaBp")

		fileOutput.write("%s\n" % refImages[i])
	 
		file = open("output.txt", "r") 
		for line in file: 
			print(line)
			fileOutput.write(line)

		fileOutput.write("\n\n")
