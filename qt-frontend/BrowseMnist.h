#pragma once

#include <QPushButton>
#include <QWidget>
#include "cublas_mlp_api.h"

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
    virtual void resizeEvent(QResizeEvent *event) override;
    void updateImg();

    CppMlpReadDims dims;
    int currentImgIdx = 0;
    QImage img;
    std::vector<float> filedata;
};
