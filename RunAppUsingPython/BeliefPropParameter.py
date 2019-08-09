from enum import Enum

class BeliefPropParameter:
	
	def __init__(self, parameter_name, parameter_value, parameter_location = BeliefPropParameterLocation.HEADER_FILE):
		self._parameter_name = parameter_name
		self._parameter_value = parameter_value
		self._parameter_location = parameter_location

	@property
	def parameter_name(self):
		return self._parameter_name

	@property
	def parameter_value(self):
		return self._parameter_value

	@property
	def parameter_location(self):
		return self._parameter_location

	@parameter_name.setter
	def parameter_name(self, value):
		self._parameter_name = value

	@parameter_value.setter
	def parameter_value(self, value):
		self._parameter_value = value

	@parameter_location.setter
	def parameter_location(self, value):
		self._parameter_location = value
