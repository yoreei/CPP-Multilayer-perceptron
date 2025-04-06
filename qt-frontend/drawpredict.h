#ifndef DRAWPREDICT_H
#define DRAWPREDICT_H

#include <thread>
#include <QWidget>

class DrawPredict : public QWidget
{
    Q_OBJECT
public:
    explicit DrawPredict(QWidget *parent = nullptr);
    ~DrawPredict() override;

private:
    void drawGrid(QPainter *painter);
    virtual void paintEvent(QPaintEvent *) override;
    QPixmap* mPixmap = nullptr;
    QImage mImage;
    float mlpInput[28*28]; // shared data
    std::thread mWorker;

    void* mlpHandle = nullptr;
    void workerThread();
};

#endif // DRAWPREDICT_H
