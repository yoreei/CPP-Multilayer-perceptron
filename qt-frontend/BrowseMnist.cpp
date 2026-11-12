#include "BrowseMnist.h"
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

    // cppmlp_read_mnist returns pixels as floats in [0, 1]

    //const char* filename = "../assets.ignored/t10k-images.idx3-ubyte";
    const char* filename = "../assets.ignored/train-images.idx3-ubyte";

    CppMlpErrorCode err = cppmlp_read_mnist_meta(filename, &dims);
    assert(err == CPPMLP_GOOD);

    filedata = std::vector<float>(dims.numImages*dims.imageCols*dims.imageRows);
    err = cppmlp_read_mnist(filename, filedata.data(), CPPMLP_READTYPE_IMAGES);
    assert(err == CPPMLP_GOOD);

    img = QImage(dims.imageCols, dims.imageRows, QImage::Format_Grayscale8);

    updateImg();

    QSize canvasSize {346,346};
    int scale = 5;
    setFixedSize(canvasSize);

    // keep the fixed size inside layouts
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
}

void BrowseBad::paintEvent(QPaintEvent *)
{
    QPainter p(this);

    // no smoothing, so the MNIST pixels stay square
    p.setRenderHint(QPainter::SmoothPixmapTransform, false);

    // scale up to the widget size, keeping the aspect ratio
    QSize target = size();
    QPixmap pm = QPixmap::fromImage(img);
    QPixmap up = pm.scaled(
        target,
        Qt::KeepAspectRatio,
        Qt::FastTransformation    // nearest neighbor
    );

    p.drawPixmap(0, 0, up);

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
