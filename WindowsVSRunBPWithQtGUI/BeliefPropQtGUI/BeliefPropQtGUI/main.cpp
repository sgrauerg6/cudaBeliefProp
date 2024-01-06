#include "BeliefPropQtGUI.h"
#include <QtWidgets/QApplication>

int main(int argc, char *argv[])
{
  QApplication a(argc, argv);
  BeliefPropQtGUI w;
  w.show();
  return a.exec();
}
