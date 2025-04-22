#ifndef TABLETCANVAS_H
#define TABLETCANVAS_H

#include <QBrush>
#include <QColor>
#include <QPen>
#include <QPixmap>
#include <QPoint>
#include <QTabletEvent>
#include <QWidget>

QT_BEGIN_NAMESPACE
class QPaintEvent;
class QString;
QT_END_NAMESPACE

class TabletCanvas : public QWidget
{
    Q_OBJECT

public:
    enum Valuator { PressureValuator, TangentialPressureValuator,
                    TiltValuator, VTiltValuator, HTiltValuator, NoValuator };
    Q_ENUM(Valuator)

    TabletCanvas(QPixmap* pixmapPtrRef, QWidget* parent);

    bool saveImage();
    void clear();
    void setAlphaChannelValuator(Valuator type)
        { m_alphaChannelValuator = type; }
    void setColorSaturationValuator(Valuator type)
        { m_colorSaturationValuator = type; }
    void setLineWidthType(Valuator type)
        { m_lineWidthValuator = type; }
    void initPixmap();
protected:
    // For tablet input
    void tabletEvent(QTabletEvent *event) override;

    // For mouse input
    void mouseMoveEvent(QMouseEvent *event) override;

    void paintEvent(QPaintEvent *event) override;

    void resizeEvent(QResizeEvent *) override
    {
        // resize in resizeEvent is a bad approach but is the easiest way to ensure
        // a pixel-specific size for our widget.
        resize(canvasSize);
    }

private:

    virtual QSize sizeHint() const override {
        return canvasSize;
    }

    // These two overloads just dont work for maintaining 1:1 aspect ratio
    // bool hasHeightForWidth() const override { return true; }
    // int heightForWidth(int w) const override { return w; }

    qreal pressureToWidth(qreal pressure);
    void paintPixmap(QPainter &painter);
    void updateBrush(const QTabletEvent *event);
    void updateBrush(const QMouseEvent *event);
    QPointF mapTo28(const QPointF& widgetPos);

    Valuator m_alphaChannelValuator = TangentialPressureValuator;
    Valuator m_colorSaturationValuator = NoValuator;
    Valuator m_lineWidthValuator = PressureValuator;
    QColor m_color = Qt::white;
    QBrush m_brush;
    QPen m_pen;
    float penSize = 2.5;
    bool mDrawing = false;
    QPixmap* mPixmapPtr;
    QSize canvasSize {346,346};

    struct Point {
        QPointF pos;
        qreal pressure = 0;
        qreal rotation = 0;
    } mLastPoint, mCurrentPoint;

signals:
    // We don't need to signal a bitmap update because MlpWorker crunches
    // constantly anyway
    // void bitmapUpdated(const QPixmap& pixmap);
};

#endif
