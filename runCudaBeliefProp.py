#!/usr/bin/env python
#make executable in bash chmod +x PyRun

import sys
import inspect
import importlib
import os

if __name__ == "__main__":
	tsukubaImageSet = {"RefImage" : "\"tsukuba1.pgm\"", "TestImage" : "\"tsukuba2.pgm\"", "CompGpuDispMap" : "\"computedDisparityMapTsukubaGPU.pgm\"", "CompCpuDispMap" : "\"computedDisparityMapTsukubaCPU.pgm\"", "NumDispVals" : "16", "ScaleBp" : "16.0f", "GroundTruthDisp" : "\"groundTruthDispTsukuba.pgm\"", "GroundTruthDispScale" : "16.0f"}	
	conesQuarterImageSet = {"RefImage" : "\"conesQuarter2.pgm\"", "TestImage" : "\"conesQuarter6.pgm\"", "CompGpuDispMap" : "\"computedDisparityConesQuarterGPU.pgm\"", "CompCpuDispMap" : "\"computedDisparityConesQuarterCPU.pgm\"", "NumDispVals" : "63", "ScaleBp" : "4.0f", "GroundTruthDisp" : "\"conesQuarterGroundTruth.pgm\"", "GroundTruthDispScale" : "4.0f"}
	conesHalfImageSet = {"RefImage" : "\"conesHalf2.pgm\"", "TestImage" : "\"conesHalf6.pgm\"", "CompGpuDispMap" : "\"computedDisparityConesHalfGPU.pgm\"", "CompCpuDispMap" : "\"computedDisparityConesQuarterCPU.pgm\"", "NumDispVals" : "90", "ScaleBp" : "2.0f", "GroundTruthDisp" : "\"conesHalfGroundTruth.pgm\"", "GroundTruthDispScale" : "2.0f"}

	imageSets = [tsukubaImageSet, conesQuarterImageSet]
	#refImages = ["\"tsukuba1.pgm\"", "\"conesQuarter2.pgm\"", "\"conesHalf2.pgm\""]
	#testImages = ["\"tsukuba2.pgm\"", "\"conesQuarter6.pgm\"", "\"conesHalf6.pgm\""]
	#saveDispGpuOutput = ["\"computedDisparityMapTsukubaGPU.pgm\"", "\"computedDisparityConesQuarterGPU.pgm\"", "\"computedDisparityConesHalfGPU.pgm\""]
	#saveDispCpuOutput = ["\"computedDisparityMapTsukubaCPU.pgm\"", "\"computedDisparityConesQuarterCPU.pgm\"", "\"computedDisparityConesHalfCPU.pgm\""]
	#numPossDispVals = ["16", "63", "90"]
	#scaleBp = ["16.0f", "4.0f", "2.0f"]
	#groundTruthDisp = ["\"groundTruthDispTsukuba.pgm\"", "\"conesQuarterGroundTruth.pgm\"", "\"conesHalfGroundTruth.pgm\""]
	#groundTruthDispScale = ["16.0f", "4.0f", "2.0f"]
	numBpLevels = ["1"]
	numBpIters = ["10"]
	truncationDiscontCost = ["1.7f"]
	truncationDataCost = ["15.0f"]
	dataCostWeight = ["0.07f"]
	smoothImagesSigma = ["0.7f"]
	fileOutput = open("outputPython.txt", "w")
	fileOutputCsv = open("outputPythonTestMoreParams2.csv", "w")
	outputLabels = []
	outputData = []
	firstLine = True
	for imageSet in imageSets:
		for currNumBpLevels in numBpLevels:
			for currNumBpIters in numBpIters:
				for currTruncationDiscontCost in truncationDiscontCost:
					for currTruncationDataCost in truncationDataCost:
						for currDataCostWeight in dataCostWeight:
							for currSmoothImagesSigma in smoothImagesSigma:
								file = open("bpParametersFromPython.h", "w")
								file.write("#ifndef BP_STEREO_FROM_PYTHON_H\n")
								file.write("#define BP_STEREO_FROM_PYTHON_H\n")
								file.write("#define REF_IMAGE_FROM_PYTHON %s\n" % imageSet["RefImage"])
								file.write("#define TEST_IMAGE_FROM_PYTHON %s\n" % imageSet["TestImage"])
								file.write("#define SAVE_DISPARITY_IMAGE_PATH_GPU_FROM_PYTHON %s\n" % imageSet["CompGpuDispMap"])
								file.write("#define SAVE_DISPARITY_IMAGE_PATH_CPU_FROM_PYTHON %s\n" % imageSet["CompCpuDispMap"])
								file.write("#define NUM_POSSIBLE_DISPARITY_VALUES_FROM_PYTHON %s\n" % imageSet["NumDispVals"])
								file.write("#define SCALE_BP_FROM_PYTHON %s\n" % imageSet["ScaleBp"])
								file.write("#define DEFAULT_GROUND_TRUTH_DISPARITY_FILE_FROM_PYTHON %s\n" % imageSet["GroundTruthDisp"])
								file.write("#define DEFAULT_GROUND_TRUTH_DISPARITY_SCALE_FROM_PYTHON %s\n" % imageSet["GroundTruthDispScale"])
								file.write("#define ITER_BP_FROM_PYTHON %s\n" % currNumBpIters)
								file.write("#define LEVELS_BP_FROM_PYTHON %s\n" % currNumBpLevels)
								file.write("#define DISC_K_BP_FROM_PYTHON %s\n" % currTruncationDiscontCost)
								file.write("#define DATA_K_BP_FROM_PYTHON %s\n" % currTruncationDataCost)
								file.write("#define LAMBDA_BP_FROM_PYTHON %s\n" % currDataCostWeight)
								file.write("#define SIGMA_BP_FROM_PYTHON %s\n" % currSmoothImagesSigma)
								file.write("#endif")
								file.close()

								os.system("make clean")
								os.system("make")
								os.system("./driverCudaBp")
 
								file = open("output.txt", "r") 
								numLabel = 0
								for line in file:
									lineSplit = line.split(":")
									if (len(lineSplit) > 0):
										if (firstLine):
											labelNoNewLine = lineSplit[0].replace("\n", "")
											outputLabels.append(labelNoNewLine)
											outputData.append([])
										if (len(lineSplit) > 1):
											dataNoNewLine = lineSplit[1].replace("\n", "")
											outputData[numLabel].append(dataNoNewLine)
										numLabel += 1			
									print(line)
									fileOutput.write(line)
				
								fileOutput.write("\n\n")
								firstLine = False

	for label in outputLabels:
		fileOutputCsv.write("%s," % label)
	fileOutputCsv.write("\n")
	for i in range(len(imageSets)*len(numBpIters)*len(numBpLevels)):
		for data in outputData:
			if (len(data) == 0):
				fileOutputCsv.write(",")
			else:
				fileOutputCsv.write("%s," % data[i])
		fileOutputCsv.write("\n")
	print outputLabels
	print outputData
	fileOutput.close()
	fileOutputCsv.close()
