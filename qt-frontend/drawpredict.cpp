#include "drawpredict.h"
#include "TabletCanvas.h"
#include <QLabel>
#include <QPushButton>
#include <QVBoxLayout>
#include <qpainter.h>
#include <QDebug>
#include "cublas_mlp_api.h"

#include <iostream>
#include <thread>
#include <chrono>
#include <atomic>

namespace {
constexpr auto SCALE = 20;
}

inline void workerThread(
        std::shared_ptr<QPixmap> mPixmap,
        std::atomic<bool>& terminate ) {
    float mlpOutput[10];
    float mlpInput[28*28];
    QImage mImage;
    void* mlpHandle = cppmlp_init("../assets.ignored");

    while (!terminate.load()) {
        // prepare data:
        int side = 28;
        mImage = mPixmap->scaled(side, side).toImage().convertToFormat(QImage::Format_Grayscale8);
        mImage.invertPixels();

        const uchar* pixelPtr = mImage.bits();
        for (int i = 0; i < side * side; ++i) {
            mlpInput[i] = (*(pixelPtr + i)) / 255.0f;
        }
        zzz drawing on the pixmap does not seem to change the return value zzz

        // predict:
        cppmlp_predict(mlpHandle, mlpInput, mlpOutput);
        for (int i = 0; i < 10; ++i) {
            std::cout << mlpOutput[i] << " ";
        }
        std::cout << std::endl;

        // (consider a condition variable)
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    // Release
    cppmlp_destroy(mlpHandle);
}

DrawPredict::DrawPredict(QWidget *parent)
    : QWidget{parent}
{
    TabletCanvas* canvas = new TabletCanvas();
    mPixmapPtr = canvas->initPixmap();
    mWorker = std::thread(workerThread, mPixmapPtr, std::ref(mTerminate));

    QVBoxLayout *layout= new QVBoxLayout;
    layout->addWidget(canvas, 1);

    QHBoxLayout *bottomLayout = new QHBoxLayout;

    // Create a widget that acts like a table with 1 header row and 1 body row.
    QWidget* tableWidget = new QWidget(this);
    QGridLayout* tableLayout = new QGridLayout(tableWidget);
    tableLayout->setSpacing(18);  // Optional: adjust spacing as needed.

    // Loop to create header (columns 0 to 9) and a corresponding body cell with a placeholder float.
    for (int col = 0; col < 10; col++) {
        // Header label with the column number.
        QLabel* headerLabel = new QLabel(QString::number(col), tableWidget);
        headerLabel->setAlignment(Qt::AlignCenter);
        tableLayout->addWidget(headerLabel, 0, col);

        // Body label with a placeholder float value.
        QLabel* bodyLabel = new QLabel(".00", tableWidget);
        bodyLabel->setAlignment(Qt::AlignCenter);
        tableLayout->addWidget(bodyLabel, 1, col);
    }
    tableWidget->setLayout(tableLayout);

    // Add the table widget to the bottom layout.
    bottomLayout->addWidget(tableWidget);

    // Add a stretch so the table stays on the left.
    bottomLayout->addStretch();

    QPushButton* clearButton = new QPushButton(tr("Clear"));
    connect(clearButton, &QPushButton::clicked, this, [canvas, this](){canvas->initPixmap(); update();});
    bottomLayout->addWidget(clearButton);

    // to the left of clearButton, in bottomLayoutWidget
    // header: from 0 to 9 horizontally
    // body: placeholder floats per col

    QWidget* bottomLayoutWidget = new QWidget(this);
    bottomLayoutWidget->setLayout(bottomLayout);
    layout->addWidget(bottomLayoutWidget, 0);

    setLayout(layout);
}

DrawPredict::~DrawPredict()
{
    mTerminate.store(true);
    // Wait for the worker thread to finish.
    mWorker.join();
}

 void DrawPredict::paintEvent(QPaintEvent *)
{
    QPainter painter(this);
    //drawGrid(&painter);
}


void DrawPredict::drawGrid(QPainter *painter)
{
    const int x = 0;
    const int y = 0;
    painter->translate(x, y);
    painter->scale(SCALE, SCALE);
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
