#include "DrawPredict.h"
#include "MlpWorker.h"
#include "TabletCanvas.h"
#include <QLabel>
#include <QPushButton>
#include <QVBoxLayout>
#include <qpainter.h>
#include <QThread>

namespace {
constexpr auto SCALE = 20;
}



DrawPredict::DrawPredict(QWidget *parent)
    : QWidget{parent}
{
    //setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Expanding);

    pixmap = std::make_unique<QPixmap>();

    mlpThread = new QThread(this);
    mlpWorker = std::make_unique<MlpWorker>(pixmap.get());
    mlpWorker->moveToThread(mlpThread); // makes the event loop of this obj run on mlpThread
    QObject::connect(mlpThread, &QThread::started, mlpWorker.get(), &MlpWorker::doWork);
    QObject::connect(mlpWorker.get(), &MlpWorker::dataUpdated, this,
                     [this](std::array<float,10> arr){
        this->updateLabels(arr);
    });

    mlpThread->start();

    // drawing canvas on top, prediction table below
    auto *mainLayout = new QVBoxLayout(this);

    setFixedWidth(28*13);
    TabletCanvas *canvas = new TabletCanvas(pixmap.get(), this);
    mainLayout->addWidget(canvas);

    // Table with the digits 0-9 in the first row and their probabilities in the second.
    QWidget* tableWidget = new QWidget(this);
    QGridLayout* tableLayout = new QGridLayout(tableWidget);
    tableLayout->setSpacing(10);

    for (int col = 0; col < 10; col++) {
        QLabel* headerLabel = new QLabel(QString::number(col), tableWidget);
        headerLabel->setAlignment(Qt::AlignCenter);
        tableLayout->addWidget(headerLabel, 0, col);

        // filled in by updateLabels() once predictions arrive
        outputLabels[col] = new QLabel(".00", tableWidget);
        outputLabels[col]->setAlignment(Qt::AlignCenter);
        tableLayout->addWidget(outputLabels[col], 1, col);
    }
    tableWidget->setLayout(tableLayout);

    mainLayout->addWidget(tableWidget, 0);

    setLayout(mainLayout);
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
