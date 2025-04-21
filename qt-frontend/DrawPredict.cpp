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

    // In your parent widget’s constructor:
    auto *mainLayout = new QVBoxLayout(this);

    setFixedWidth(28*13);
    TabletCanvas *canvas = new TabletCanvas(pixmap.get(), this);
    mainLayout->addWidget(canvas);

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

    QVBoxLayout* bottomGroup = new QVBoxLayout;
    bottomGroup->addWidget(tableWidget);

    // now add that group into your bottomLayout, then the stretch:
    bottomLayout->addLayout(bottomGroup);

    // to the left of clearButton, in bottomLayoutWidget
    // header: from 0 to 9 horizontally
    // body: placeholder floats per col

    QWidget* bottomLayoutWidget = new QWidget(this);
    bottomLayoutWidget->setLayout(bottomLayout);
    mainLayout->addWidget(bottomLayoutWidget, 0);
    mainLayout->addStretch();

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
