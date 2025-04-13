#include "DrawPredict.h"
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
#include <span>

namespace {
constexpr auto SCALE = 20;
}



DrawPredict::DrawPredict(QWidget *parent)
    : QWidget{parent}
{

    qDebug() << "DrawPredict";
    mPixmap = std::make_unique<QPixmap>();
    TabletCanvas* canvas = new TabletCanvas(mPixmap.get());
    mWorker = std::thread(workerThread, mPixmap.get(), std::ref(mTerminate));

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
        QLabel* bodyLabel = new QLabel(".00", tableWidget); // this label should reflect the value of mMlpOutput[col]
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
    drawGrid(&painter);
}


void DrawPredict::drawGrid(QPainter *painter)
{
    QString debugOutput = "";
    for (int i = 0; i < 10; ++i) {
        debugOutput += QString::number(output[i]) + " ";
    }
    qDebug() << "mlpOutput: " << debugOutput;

}
