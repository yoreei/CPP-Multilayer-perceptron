
// Copyright (C) 2016 The Qt Company Ltd.
// SPDX-License-Identifier: LicenseRef-Qt-Commercial OR BSD-3-Clause

#ifndef TABLETCANVAS_H
#define TABLETCANVAS_H

#include <QBrush>
#include <QColor>
#include <QPen>
#include <QPixmap>
#include <QPoint>
#include <QTabletEvent>
#include <QWidget>
#include <memory>

QT_BEGIN_NAMESPACE
class QPaintEvent;
class QString;
QT_END_NAMESPACE

//! [0]
class TabletCanvas : public QWidget
{
    Q_OBJECT

public:
    enum Valuator { PressureValuator, TangentialPressureValuator,
                    TiltValuator, VTiltValuator, HTiltValuator, NoValuator };
    Q_ENUM(Valuator)

    TabletCanvas();

    bool saveImage();
    //bool loadImage(const QString &file);
    void clear();
    void setAlphaChannelValuator(Valuator type)
        { m_alphaChannelValuator = type; }
    void setColorSaturationValuator(Valuator type)
        { m_colorSaturationValuator = type; }
    void setLineWidthType(Valuator type)
        { m_lineWidthValuator = type; }
    void setColor(const QColor &c)
        { if (c.isValid()) m_color = c; }
    QColor color() const
        { return m_color; }
    std::shared_ptr<QPixmap> initPixmap();
protected:
    void tabletEvent(QTabletEvent *event) override;
    void paintEvent(QPaintEvent *event) override;
    void resizeEvent(QResizeEvent *event) override;

private:
    void paintPixmap(QPainter &painter, QTabletEvent *event);
    qreal pressureToWidth(qreal pressure);
    void updateBrush(const QTabletEvent *event);
    //void updateCursor(const QTabletEvent *event);

    Valuator m_alphaChannelValuator = TangentialPressureValuator;
    Valuator m_colorSaturationValuator = NoValuator;
    Valuator m_lineWidthValuator = PressureValuator;
    QColor m_color = Qt::black;
    QBrush m_brush;
    QPen m_pen;
    float penSize = 12;
    bool m_deviceDown = false;
    std::shared_ptr<QPixmap> mPixmapPtr = nullptr;

    struct Point {
        QPointF pos;
        qreal pressure = 0;
        qreal rotation = 0;
    } lastPoint;

signals:
    // Declare a signal that can be emitted when needed
    void bitmapUpdated(const QPixmap& pixmap);
};
//! [0]

#endif
