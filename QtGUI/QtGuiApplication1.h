#pragma once

#include <QtWidgets/QMainWindow>
#include "ui_QtGuiApplication1.h"
#include <QPixmap>
#include "GuiProcessStereoSet.h"
#include "bpStereoParameters.h"

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
