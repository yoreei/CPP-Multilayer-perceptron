#include "Tab.h"
#include <QApplication>

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);

    // Uncomment this to debug layout:
    // a.setStyleSheet(R"(
    //     QWidget {
    //         border: 1px dashed rgba(255,0,0,0.5);
    //     }
    // )");

    Tab t;
    t.show();
    return a.exec();
}
