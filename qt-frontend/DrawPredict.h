#ifndef DRAWPREDICT_H
#define DRAWPREDICT_H

#include <thread>
#include <QWidget>
#include <QLabel>

class DrawPredict : public QWidget
{
    Q_OBJECT
public:
    explicit DrawPredict(QWidget *parent = nullptr);
    ~DrawPredict() override;

private:
    void drawGrid(QPainter *painter);
    virtual void paintEvent(QPaintEvent *) override;

    std::array<QLabel*, 10> mOutputLabels;
    // worker thread stuff
    std::array<float, 10> mMlpOutput;
    std::thread mWorker;
    std::unique_ptr<QPixmap> mPixmap = nullptr;
    std::atomic<bool> mTerminate{false};  // workerThread exit signal
};

#endif // DRAWPREDICT_H
