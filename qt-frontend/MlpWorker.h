#ifndef MLPWORKER_H
#define MLPWORKER_H

#include <QImage>
#include <QObject>
#include <array>
#include <atomic>

class QPixmap;
class MlpWorker : public QObject {
    Q_OBJECT
public:
    explicit MlpWorker(
        QPixmap* pixmapPtr,
        QObject *parent = nullptr) :
        QObject(parent),
        pixmapPtr(pixmapPtr) {}

signals:
    void dataUpdated(const std::array<float, 10> output);

public slots:
    void doWork();

private:
QPixmap* pixmapPtr = nullptr;
};

#endif // MLPWORKER_H
