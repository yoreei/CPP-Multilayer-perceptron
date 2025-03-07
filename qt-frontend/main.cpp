//#include "BrowseBad.h"
#include "Tab.h"
#include <QApplication>

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    // BrowseBad w;
    // w.show();
    Tab t;
    t.show();
    return a.exec();
}
