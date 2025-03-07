#pragma once

#include <QPushButton>
#include <QWidget>
#include <memory>

namespace Ui {
class Example;
}

class BrowseBad : public QWidget
{
    Q_OBJECT

public:
    BrowseBad(QWidget *parent = nullptr);
    ~BrowseBad();

protected:
    void paintEvent(QPaintEvent *) override;
    void wheelEvent(QWheelEvent *) override;
    void nextStep(int delta);
    //void drawPoint(QPainter *painter, const QPointF &pos, const QColor &color);
    virtual void resizeEvent(QResizeEvent *event) override;

    int m_step = 0;
    int numBad = 0;
    QPushButton* prev = new QPushButton{"<"};
    QPushButton* next = new QPushButton{">"};
};
