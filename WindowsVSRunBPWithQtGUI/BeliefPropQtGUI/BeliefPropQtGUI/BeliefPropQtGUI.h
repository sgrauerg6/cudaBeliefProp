#pragma once

#include <QtWidgets/QMainWindow>
#include "ui_BeliefPropQtGUI.h"
#include <QPixmap>
#include "GuiProcessStereoSet.h"
//needed for the current BP parameters for the costs and also the CUDA parameters such as thread block size
#include "./BpConstsAndParams/bpStereoParameters.h"
#include "./BpConstsAndParams/RunSettings.h"
#include "./BpConstsAndParams/bpStructsAndEnums.h"

//needed to run the each of the bp implementations
#include "BpSingleThreadCPU/stereo.h"
#include "BpOptimizeCPU/RunBpStereoOptimizedCPU.h"
#include "BpOptimizeCUDA/RunBpStereoSetOnGPUWithCUDA.h"
#include "./BpFileProcessing/BpFileHandling.h"
#include "./BpConstsAndParams/RunSettings.h"
#include "./BpRunProcessing/RunBpStereoSet.h"

class BeliefPropQtGUI : public QMainWindow
{
  Q_OBJECT

public:
  BeliefPropQtGUI(QWidget* parent = Q_NULLPTR);
  void processButtonPress();

private:
  Ui::BeliefPropQtGUIClass ui;
  bool stereo_processing_run_;
};


/*#pragma once

#include <QtWidgets/QMainWindow>
#include "ui_BeliefPropQtGUI.h"

class BeliefPropQtGUI : public QMainWindow
{
  Q_OBJECT

public:
  BeliefPropQtGUI(QWidget *parent = Q_NULLPTR);

private:
  Ui::BeliefPropQtGUIClass ui;
};*/
