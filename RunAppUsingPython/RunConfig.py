from RunConfigParameterInfoAndDefaults import *

class RunConfig:

	def __init__(self):
		self._parameters = {}
		for param in RunConfigParams:
			self._parameters[param] = BeliefPropParameter(RunConfigParameterLabels.RUN_CONFIG_PARAMS_LABELS[param], RunConfigDefaultValues.RUN_CONFIG_PARAMS_DEFAULT_VALUES[param], RunConfigParameterLocations.RUN_CONFIG_PARAMS_LOCATIONS[param])

	def updateBpParamsTextFileWithCurrentConfig(self):
		file = open("bpParametersFromPython.txt", "w")
		for param in self._parameters:
			if (param.parameter_location() == BeliefPropParameterLocation.PARAMETERS_TEXT_FILE):
				file.write(param.parameter_name() + " : " + str(param.parameter_value()))
		file.close()

	def updateBpParamsHeaderFileWithCurrentConfig(self):
		file = open("bpParametersFromPython.h", "w")
		file.write("#ifndef BP_STEREO_FROM_PYTHON_H\n")
		file.write("#define BP_STEREO_FROM_PYTHON_H\n")
		for param in self._parameters:
			if (param.parameter_location() == BeliefPropParameterLocation.HEADER_FILE):
				file.write("#define " + param.parameter_name() + " " + str(param.parameter_value()))
		file.write("#endif")
		file.close()

	def getConfigParameterValue(self, parameter):
		return self._parameters[param].parameter_value()
