#!/usr/bin/env python
#make executable in bash chmod +x PyRun

from RunConfig import RunConfig
from MakeAndRunBeliefProp import MakeAndRunBeliefProp
from ProcessBeliefPropOutput import ProcessBeliedPropOutput

if __name__ == "__main__":

	currRunConfig = RunConfig()
	currRunConfig.updateBpParamsFileWithCurrentConfig()
	#currConfigsToRun = MultConfigs()
	#runConfigs = currConfigsToRun.retrieveAllConfigs()
	currProcessBeliefPropOutput = ProcessBeliefPropOutput("output.txt")
	currMakeAndRunBeliefProp = MakeAndRunBeliefProp(currRunConfig, "../", "../")
	currMakeAndRunBeliefProp(currRunConfig, "../", "./../driverCudaBp")
	currProcessBeliefPropOutput.processOutputCurrentRun("output.txt")
	currProcessBeliefPropOutput.processFinalResultsWriteToFile(1)

