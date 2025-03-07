#include "drawpredict.h"
#include <qpainter.h>

namespace {
constexpr auto SCALE = 20;
}

DrawPredict::DrawPredict(QWidget *parent)
    : QWidget{parent}
{}

 void DrawPredict::paintEvent(QPaintEvent *)
{
    QPainter painter(this);

    const int x = 0; //ui->layout->contentsMargins().left();
    const int y = 0;
    painter.translate(x, y);
    painter.scale(SCALE, SCALE);

    drawGrid(&painter);
}


void DrawPredict::drawGrid(QPainter *painter)
{
    int numLinesW = width() / SCALE - 1;
    int numLinesH = height() / SCALE;// ui->scratchHeight->geometry().height() / SCALE ;
    auto setPen = [&painter](int x, int endLine) {
        if (x % 5 == 0 || x == endLine)
            painter->setPen(QPen(Qt::gray, 0, Qt::SolidLine));
        else
            painter->setPen(QPen(Qt::lightGray, 0, Qt::DotLine));
    };

    for (int x = 0; x <= numLinesW; x += 1) {
        setPen(x, numLinesW);
        painter->drawLine(x, 0, x, numLinesH);
    }

    for (int y = 0; y <= numLinesH; y += 1) {
        setPen(y, numLinesH);
        painter->drawLine(0, y, numLinesW, y);
    }
}
