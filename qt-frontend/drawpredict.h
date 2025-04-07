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

    // worker thread stuff
    std::thread mWorker;
    std::shared_ptr<QPixmap> mPixmapPtr = nullptr;
    std::atomic<bool> mTerminate{false};  // workerThread exit signal
};

#endif // DRAWPREDICT_H
