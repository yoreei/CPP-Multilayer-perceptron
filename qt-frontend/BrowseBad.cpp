#include "BrowseBad.h"
#include <QFontMetricsF>
#include <QPainter>
#include <QVBoxLayout>
#include <QWheelEvent>

BrowseBad::BrowseBad(QWidget *parent)
    : QWidget(parent)
{
    // ui->setupUi(this);
    // connect(ui->prev, &QToolButton::clicked, this, [this] { nextStep(-1); });
    // connect(ui->next, &QToolButton::clicked, this, [this] { nextStep(+1); });

    // assume `float image[28*28]` has values in [0,1] (or [0,255] – see below)
    float image[28*28]
        zzz get image from my dll

    QImage img(28, 28, QImage::Format_Grayscale8);

    for (int y = 0; y < 28; ++y) {
        for (int x = 0; x < 28; ++x) {
            float v = image[y*28 + x];
            // if your data is 0..1:
            int gray = qBound(0, int(v * 255.0f + 0.5f), 255);
            // if it’s already 0..255, just cast to int+clamp
            img.setPixel(x, y, gray);
        }
    }


    QVBoxLayout *layout= new QVBoxLayout;
    setLayout(layout);

    nextStep(0);
}

void BrowseBad::paintEvent(QPaintEvent *)
{

}

BrowseBad::~BrowseBad() = default;

void BrowseBad::wheelEvent(QWheelEvent *event)
{
    nextStep(event->angleDelta().y() > 0 ? -1 : 1);
}

void BrowseBad::resizeEvent(QResizeEvent *event)
{
    qDebug() << QString("I was resized %1x%2").arg(width()).arg(height());
}


// void BrowseBad::drawPoint(QPainter *painter, const QPointF &pos, const QColor &color)
// {
//     painter->save();
//     painter->setPen(Qt::NoPen);
//     painter->setBrush(color);
//     painter->drawEllipse(pos, 10.0 / SCALE, 10.0 / SCALE);

//     //setItalic(painter, false);

//     painter->translate(pos);
//     painter->scale(2.0 / SCALE, 2.0 / SCALE);
//     painter->translate(-pos);

//     painter->translate(5, -5);
//     painter->setPen(color);
//     painter->drawText(pos, QString("(%1,%2)").arg(pos.x()).arg(pos.y()));
//     painter->restore();
// }

void BrowseBad::nextStep(int delta)
{
    m_step += delta;

    //ui->description->setText(title);
    update();
    prev->setEnabled(m_step > 0);
    next->setEnabled(m_step < numBad);
}
