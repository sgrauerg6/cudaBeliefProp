import os
from RunConfig import RunConfig
from RunConfigParameterInfoAndDefaults import RunConfigParams

class MakeAndRunBeliefProp:

	def __init__(self, currConfig, makefileDir, exeFilePath):
		self._currConfig = currConfig
		self._makefileDir = makefileDir
		self._exeFilePath = exeFilePath

	def __clean(self):
		os.system("make clean -C " + self._makefileDir)

	def __setParamsAndMakeBuild(self):
		os.environ["OMP_NUM_THREADS"] = self._currConfig.getConfigParameterValue(RunConfigParams.NUM_OPENMP_THREADS)
		os.system("make -C " + self._makefileDir)

	def __call__(self):
		self.__clean()
		self.__setParamsAndMakeBuild()
		os.system(self._exeFilePath)
