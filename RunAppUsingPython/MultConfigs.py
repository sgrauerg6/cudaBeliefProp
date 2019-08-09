class MultConfigs:

	def __init__(self):
		self._imageSets = [conesImageSetQuarterSize]	
		self._numBpIters = [7]
		self._truncationDiscontCost = [1.7]
		self._truncationDataCost = [15.0]
		self._dataCostWeight = [0.1]
		self._smoothImagesSigma = [0.0]
		self._fileOutput = open("outputPython.txt", "w")
		self._optimizedMemory = [1]
		self._beliefPropDataTypeProcessing = [0]
		self._indexOptimizationSettings = [1]
		self._cpuOptimizationSettings = [0]
		self._memoryAlignmentOptimizations = [1]
		self._startRegisterMemoryOptions = [0]
		self._localMemoryOptions = [0]
		self._numDispValsSet = [128]

	def getMultConfigsFromFile(self, filePath):
		

	def retrieveAllConfigs(self):
		configs = []
		for imageSet in self._imageSets:
			for currNumBpLevelsAndIters in self._numBpLevelsAndIters:
				for currTruncationDiscontCost in self._truncationDiscontCost:
					for currTruncationDataCost in self._truncationDataCost:
						for currDataCostWeight in self._dataCostWeight:
							for currSmoothImagesSigma in self._smoothImagesSigma:
								for optMemory in self._optimizedMemory:
									for beliefPropDataTypeToProcess in self._beliefPropDataTypeProcessing:
										for currentIndexOptimizationSettings in self._indexOptimizationSettings:
											for cpuOptimizationSetting in self._cpuOptimizationSettings:
												for currNumOpenMPThreads in self._numOpenMPThreads:
													for memoryAlignmentOptimization in self._memoryAlignmentOptimizations:
														currConfig = RunConfig()
														currConfig.setImageSet(imageSet)
														currConfig.setBpLevels(bpLevels)
														currConfig.setBpIters(bpIters)
														currConfig.setTruncationDiscontCost(currTruncationDiscontCost)
														currConfig.setTruncationDataCost(currTruncationDataCost)
														currConfig.setDataCostWeight(currDataCostWeight)
														currConfig.setSmoothImagesSigma(currSmoothImagesSigma)
														currConfig.setOptMemory(optMemory)
														currConfig.setDataTypeToProcess(beliefPropDataTypeToProcess)
														currConfig.setIndexOptimizationSettings(currentIndexOptimizationSettings)
														currConfig.setCpuOptimizationSetting(cpuOptimizationSetting)
														currConfig.setNumOpenMPThreads(currNumOpenMPThreads)
														currConfig.setMemoryAlignmentOptimzation(memoryAlignmentOptimization)
														configs.append(currConfig)

		return configs


		
