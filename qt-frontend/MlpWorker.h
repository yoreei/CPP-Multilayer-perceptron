#ifndef MLPWORKER_H
#define MLPWORKER_H

#include <QObject>

class MlpWorker : public QObject {
    Q_OBJECT
public:
    explicit MlpWorker(
        QPixmap* pixmapPtr,
        std::array<float, 10>* output,
        std::atomic<bool>* terminate,
        QObject *parent = nullptr) :
        QObject(parent),
        pixmapPtr(pixmapPtr),
        output(output),
        terminate(terminate)    {}

signals:
    void dataUpdated(const QString &newData);

public slots:
    void doWork() {
    assert(pixmapPtr);
    float mlpInput[28*28];
    QImage mImage;
    CppMlpHndl mlpHndl;
    qDebug() << "initializing cppmlp";
    CppMlpErrorCode err = cppmlp_init(&mlpHndl, "../assets.ignored");
    if(err != CPPMLP_GOOD) {
        throw std::runtime_error("cppmlp_init");
    }
    qDebug() << "cppmlp initialized";

    qDebug() << "workerThread: mPixmapPtr " << pixmapPtr;
    while (!terminate.load()) {
        // prepare data:
        int side = 28;
        mImage = pixmapPtr->scaled(side, side).toImage().convertToFormat(QImage::Format_Grayscale8);
        mImage.invertPixels();

        const uchar* pixelPtr = mImage.bits();
        //qDebug() << "workerThread: 1st pixel " << QString::number(*pixelPtr);
        QString mlpInputString = "";
        for (int i = 0; i < side * side; ++i) {
            mlpInput[i] = (*(pixelPtr + i)) / 255.0f;
            //mlpInputString += QString::number(mlpInput[i]) + " ";
        }
        //qDebug() << "mlpInputString: " << mlpInputString;
        //zzz drawing on the pixmap does not seem to change the return value zzz

        // predict:
        CppMlpErrorCode errPredict = cppmlp_predict(mlpHndl, mlpInput, output);
        if(errPredict != CPPMLP_GOOD) {
            throw std::runtime_error("cppmlp_predict");
        }

        emit dataUpdated(result);

        // (consider a condition variable)
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    // Release
    cppmlp_destroy(mlpHndl);
    }

private:
QPixmap* pixmapPtr = nullptr;
std::array<float, 10>*     float (&output)[10];
std::atomic<bool>* terminate;
};

#endif // MLPWORKER_H
