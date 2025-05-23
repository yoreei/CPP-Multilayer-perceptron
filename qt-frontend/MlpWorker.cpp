#include "MlpWorker.h"
#include "cublas_mlp_api.h"
#include <QDebug>
#include <QPixmap>
#include <QThread>

void MlpWorker::doWork() {
    assert(pixmapPtr);

    // `side` is hardcoded for the MNIST database
    const int side = 28;

    float mlpInput[side*side];
    QImage mImage;
    CppMlpHndl mlpHndl;
    qDebug() << "initializing cppmlp";
    CppMlpErrorCode err = cppmlp_init(&mlpHndl, "../assets.ignored");
    if(err != CPPMLP_GOOD) {
        throw std::runtime_error("cppmlp_init");
    }
    qDebug() << "cppmlp initialized";

    qDebug() << "workerThread: mPixmapPtr " << pixmapPtr;
    while (!QThread::currentThread()->isInterruptionRequested()) {
        // prepare data:
        mImage = pixmapPtr->toImage().convertToFormat(QImage::Format_Grayscale8);

        const uchar* pixelPtr = mImage.bits();
        QString mlpInputString = "";
        for (int i = 0; i < side * side; ++i) {
            mlpInput[i] = (*(pixelPtr + i)) / 255.0f;
        }

        // predict:
        std::array<float, 10> output;
        CppMlpErrorCode errPredict = cppmlp_predict(mlpHndl, mlpInput, output.data());
        if(errPredict != CPPMLP_GOOD) {
            throw std::runtime_error("cppmlp_predict");
        }

        emit dataUpdated(output);

        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    // Release
    cppmlp_destroy(mlpHndl);
}
