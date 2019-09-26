#pragma once

#include <QtWidgets/QMainWindow>
#include "ui_QtGuiApplication1.h"
#include <QPixmap>
#include "GuiProcessStereoSet.h"
//needed for the current BP parameters for the costs and also the CUDA parameters such as thread block size
#include "./ParameterFiles/bpStereoParameters.h"
#include "./ParameterFiles/bpRunSettings.h"
#include "./ParameterFiles/bpStructsAndEnums.h"

//needed to run the each of the bp implementations
#include "SingleThreadCPU/stereo.h"
#include "OptimizeCPU/RunBpStereoOptimizedCPU.h"
#include "OptimizeCUDA/RunBpStereoSetOnGPUWithCUDA.h"
#include "./FileProcessing/BpFileHandling.h"
#include "./ParameterFiles/bpRunSettings.h"
#include "./BpAndSmoothProcessing/RunBpStereoSet.h"

class QtGuiApplication1 : public QMainWindow
{
	Q_OBJECT

public:
	QtGuiApplication1(QWidget *parent = Q_NULLPTR);
	void processButtonPress();

private:
	Ui::QtGuiApplication1Class ui;
	bool stereo_processing_run_;
};
