#include "DrawPredict.h"
#include "MlpWorker.h"
#include "TabletCanvas.h"
#include <QLabel>
#include <QPushButton>
#include <QVBoxLayout>
#include <qpainter.h>
#include <QDebug>
#include <QThread>

namespace {
constexpr auto SCALE = 20;
}



DrawPredict::DrawPredict(QWidget *parent)
    : QWidget{parent}
{

    qDebug() << "DrawPredict";
    pixmap = std::make_unique<QPixmap>();
    TabletCanvas* canvas = new TabletCanvas(pixmap.get());
    canvas->setFixedSize(500, 500);

    mlpThread = new QThread(this);
    mlpWorker = std::make_unique<MlpWorker>(pixmap.get());
    mlpWorker->moveToThread(mlpThread); // makes the event loop of this obj run on mlpThread
    QObject::connect(mlpThread, &QThread::started, mlpWorker.get(), &MlpWorker::doWork);
    QObject::connect(mlpWorker.get(), &MlpWorker::dataUpdated, this,
                     [this](std::array<float,10> arr){
        this->updateLabels(arr);
    });

    mlpThread->start();

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
        outputLabels[col] = new QLabel(".00", tableWidget);
        outputLabels[col]->setAlignment(Qt::AlignCenter);
        tableLayout->addWidget(outputLabels[col], 1, col);
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
    mlpThread->requestInterruption();
    mlpThread->quit();
    mlpThread->wait();
    // mlpThread will destruct mlpWorker using signals
}

void DrawPredict::updateLabels(std::array<float, 10> output)
{
    for(int i = 0; i < 10; ++i) {
        assert(outputLabels[i]);
        outputLabels[i]->setText(QString::number(output[i], 'f', 2));
    }
    update();
}
