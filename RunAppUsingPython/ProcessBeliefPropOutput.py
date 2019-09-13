import time

class ProcessBeliefPropOutput:

	def __init__(self, filePathBeliefPropFileOutput):
		self._outputLabels = []
		self._outputData = []
		self._outputFileOutputCsvName = self._getOutputFileNameWithTime()
		self._runBeliefPropResultsFileOutput = open(filePathBeliefPropFileOutput, "w")

	def _getOutputFileNameWithTime(self):
		currTime = time.time()
		self.outputFileOutputCsvName = "outputPythonTestBenchmarkSets" + str(currTime) + ".csv"

	def processOutputCurrentRun(self):
		numLabel = 0
		for line in file:
			lineSplit = line.split(":")
			if (len(lineSplit) > 0):
				if (firstLine):
					labelNoNewLine = lineSplit[0].replace("\n", "")
					self._outputLabels.append(labelNoNewLine)
					self._outputData.append([])
				if (len(lineSplit) > 1):
					dataNoNewLine = lineSplit[1].replace("\n", "")
					self._outputData[numLabel].append(dataNoNewLine)
					numLabel += 1			
				print(line)
				self._runBeliefPropResultsFileOutput.write(line)
								
		self._runBeliefPropResultsFileOutput.write("\n\n")
		firstLine = False

	def processFinalResultsWriteToFile(self, numRuns):
		# close the run belief prop file output since done retrieving results from runs
		self._runBeliefPropResultsFileOutput.close()

		self._runBeliefPropFinalResultsCsvFileOutput = open(self.outputFileOutputCsvName, "w")
		for label in self._outputLabels:
			self._runBeliefPropFinalResultsCsvFileOutput.write("%s," % label)
		self._runBeliefPropFinalResultsCsvFileOutput.write("\n")

		for i in range(numRuns):
			for data in self._outputData:
				if (len(data) == 0):
					self._runBeliefPropFinalResultsCsvFileOutput.write(",")
				else:
					self._runBeliefPropFinalResultsCsvFileOutput.write("%s," % data[i])
			self._runBeliefPropFinalResultsCsvFileOutput.write("\n")
		print self._outputLabels
		print self._outputData
		self._runBeliefPropFinalResultsCsvFileOutput.close()
		
