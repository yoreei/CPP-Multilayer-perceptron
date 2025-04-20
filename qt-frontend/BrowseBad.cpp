#include "BrowseBad.h"
#include <QFontMetricsF>
#include <QPainter>
#include <QVBoxLayout>
#include <QWheelEvent>
#include "cublas_mlp_api.h"

BrowseBad::BrowseBad(QWidget *parent)
    : QWidget(parent)
{
    // ui->setupUi(this);
    // connect(ui->prev, &QToolButton::clicked, this, [this] { nextStep(-1); });
    // connect(ui->next, &QToolButton::clicked, this, [this] { nextStep(+1); });

    // assume `float image[28*28]` has values in [0,1] (or [0,255] – see below)

    //const char* filename = "../assets.ignored/t10k-images.idx3-ubyte";
    const char* filename = "../assets.ignored/train-images.idx3-ubyte";

    CppMlpErrorCode err = cppmlp_read_mnist_meta(filename, &dims);
    assert(err == CPPMLP_GOOD);

    filedata = std::vector<float>(dims.numImages*dims.imageCols*dims.imageRows);
    err = cppmlp_read_mnist(filename, filedata.data(), CPPMLP_READTYPE_IMAGES);
    assert(err == CPPMLP_GOOD);

    auto maxElem = std::max_element(filedata.begin(), filedata.end());
    qDebug() << "global max elem:" << *maxElem << "\n";

    img = QImage(dims.imageCols, dims.imageRows, QImage::Format_Grayscale8);

    updateImg();

    int scale = 15;
    QSize fixed = QSize(dims.imageCols * scale, dims.imageRows * scale);
    setFixedSize(fixed);

    // ensure layouts won’t stretch it:
    setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);
}

void BrowseBad::updateImg(){
    int skip = currentImgIdx * dims.imageRows * dims.imageCols;
    for (int y = 0; y < dims.imageRows; ++y) {
        for (int x = 0; x < dims.imageCols; ++x) {
            float v = filedata[skip + y*dims.imageCols + x];
            int gray = qBound(0, int(v * 255.0f), 255);
            QColor c {gray, gray, gray};
            img.setPixelColor(x, y, c);
        }
    }

    int skip2 = (currentImgIdx + 1) * dims.imageRows * dims.imageCols;
    auto maxElem = std::max_element(filedata.begin() + skip, filedata.begin() + skip2);
    qDebug() << "current max elem:" << *maxElem << "\n";

    qDebug() << "image:";
    QString str="";
    for (auto i = filedata.begin() + skip; i < filedata.begin() + skip2; ++ i){
        str = str + QString::number(*i) + " ";
    }
    qDebug() << str;
}

void BrowseBad::paintEvent(QPaintEvent *)
{
    QPainter p(this);

    // turn OFF smooth pixmap transforms
    p.setRenderHint(QPainter::SmoothPixmapTransform, false);

    // compute a block‑y scale to fill the widget, preserving aspect
    QSize target = size();
    QPixmap pm = QPixmap::fromImage(img);
    QPixmap up = pm.scaled(
        target,
        Qt::KeepAspectRatio,
        Qt::FastTransformation    // <-- nearest‑neighbor
    );

    // center it
    int x = (width()  - up.width())  / 2;
    int y = (height() - up.height()) / 2;
    p.drawPixmap(x, y, up);

}

BrowseBad::~BrowseBad() = default;

void BrowseBad::wheelEvent(QWheelEvent *event)
{
    if(event->angleDelta().y() > 0) {
        currentImgIdx = std::clamp(currentImgIdx + 1, 0, dims.numImages);
    }
    else {
        currentImgIdx = std::clamp(currentImgIdx - 1, 0, dims.numImages);
    }
    updateImg();
    update();
}

void BrowseBad::resizeEvent(QResizeEvent *event)
{
    qDebug() << QString("I was resized %1x%2").arg(width()).arg(height());
}
