#ifndef DRAWPREDICT_H
#define DRAWPREDICT_H

#include <QWidget>

class DrawPredict : public QWidget
{
    Q_OBJECT
public:
    explicit DrawPredict(QWidget *parent = nullptr);

private slots:
    // Slot that handles the signal from TabletCanvas
    void onBitmapUpdated();

private:
    void drawGrid(QPainter *painter);
    virtual void paintEvent(QPaintEvent *) override;
};

#endif // DRAWPREDICT_H
