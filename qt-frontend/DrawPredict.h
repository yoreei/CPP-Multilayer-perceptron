#ifndef DRAWPREDICT_H
#define DRAWPREDICT_H

#include <QWidget>
#include <QLabel>
#include <QThread>

class MlpWorker;

class DrawPredict : public QWidget
{
    Q_OBJECT
public:
    explicit DrawPredict(QWidget *parent = nullptr);
    ~DrawPredict() override;

private:
    void updateLabels(std::array<float, 10> output);

    std::array<QLabel*, 10> outputLabels{};
    // worker thread stuff
    QThread* mlpThread = nullptr;
    std::unique_ptr<MlpWorker> mlpWorker = nullptr;
    std::unique_ptr<QPixmap> pixmap = nullptr;
};

#endif // DRAWPREDICT_H
