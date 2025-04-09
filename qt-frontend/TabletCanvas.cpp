
// Copyright (C) 2016 The Qt Company Ltd.
// SPDX-License-Identifier: LicenseRef-Qt-Commercial OR BSD-3-Clause

#include "TabletCanvas.h"

#include <QCoreApplication>
#include <QPainter>
#include <QShortcut>
#include <QtMath>
#include <cstdlib>

//! [0]
TabletCanvas::TabletCanvas()
    : QWidget(nullptr), m_brush(m_color)
    , m_pen(m_brush, 1, Qt::SolidLine, Qt::RoundCap, Qt::RoundJoin)
{
    resize(500, 500);
    setAttribute(Qt::WA_TabletTracking); // see docs for `tabletTracking`

    QShortcut* shortcut = new QShortcut(QKeySequence("Ctrl+S"), this);
    connect(shortcut, &QShortcut::activated, this, &TabletCanvas::saveImage);

}
//! [0]

//! [1]
bool TabletCanvas::saveImage()
{
    return mPixmapPtr->scaled(28, 28).save("C:\\DATA\\git\\cpp-mlp\\assets.ignored\\tabletOut.png");
}

// bool TabletCanvas::loadImage(const QString &file)
// {
//     bool success = m_pixmap.load(file);

//     if (success) {
//         update();
//         return true;
//     }
//     return false;
// }

void TabletCanvas::clear()
{
    mPixmapPtr->fill(Qt::white);
    update();
}

void TabletCanvas::tabletEvent(QTabletEvent *event)
{
    switch (event->type()) {
    case QEvent::TabletPress:
        mDrawing = true;
        break;
    case QEvent::TabletMove:
//#ifndef Q_OS_IOS
        //if (event->pointingDevice() && event->pointingDevice()->capabilities().testFlag(QPointingDevice::Capability::Rotation))
        //updateCursor(event);
//#endif
        if (mDrawing) {
            updateBrush(event);
            QPainter painter(mPixmapPtr.get());
            paintPixmap(painter, event);
            // emit bitmapUpdated(mPixmapPtr.get()); // worker thread crunches constantly, no need to emit
        }
        break;
        case QEvent::TabletRelease:
            mDrawing = false;
            break;
        default:
            break;
        }
    mLastPoint.pos = event->position();
    mLastPoint.pressure = event->pressure();
    mLastPoint.rotation = event->rotation();
    update();
    event->accept();
}

void TabletCanvas::mousePressEvent(QMouseEvent *event)
{
    if (event->button() == Qt::LeftButton) {
        mDrawing = true;
        mLastPoint.pos = event->position();
    }
}

void TabletCanvas::mouseMoveEvent(QMouseEvent *event)
{
    if (mDrawing && (event->buttons() & Qt::LeftButton)) {
        updateBrush(event);
        QPainter painter(mPixmapPtr.get());
        paintPixmap(painter, event);
        mLastPoint.pos = event->pos();
        update();
    }
}

void TabletCanvas::mouseReleaseEvent(QMouseEvent *event)
{
    if (event->button() == Qt::LeftButton && mDrawing) {
        mDrawing = false;
        update();
    }
}


std::shared_ptr<QPixmap> TabletCanvas::initPixmap()
{
    qreal dpr = devicePixelRatio();
    mPixmapPtr = std::make_shared<QPixmap>(qRound(width() * dpr), qRound(height() * dpr));
    mPixmapPtr->setDevicePixelRatio(dpr);
    mPixmapPtr->fill(Qt::white);
    return mPixmapPtr;
}

void TabletCanvas::paintEvent(QPaintEvent *event)
{
    assert(mPixmapPtr);
    QPainter painter(this);
    QRect pixmapPortion = QRect(event->rect().topLeft() * devicePixelRatio(),
                                event->rect().size() * devicePixelRatio());
    painter.drawPixmap(event->rect().topLeft(), *mPixmapPtr, pixmapPortion);
}
//! [4]
//! [5]
void TabletCanvas::paintPixmap(QPainter &painter, QTabletEvent *event)
{
    painter.setRenderHint(QPainter::Antialiasing);

    auto deviceType = event->deviceType();
    // mouse and puck should not even land in this tablet specific-code
    assert(deviceType != QInputDevice::DeviceType::Mouse && deviceType != QInputDevice::DeviceType::Puck);
    deviceType = QInputDevice::DeviceType::Airbrush;
    switch (deviceType) {
//! [6]
        case QInputDevice::DeviceType::Airbrush:
            {
                painter.setPen(Qt::NoPen);
                QRadialGradient grad(mLastPoint.pos, m_pen.widthF() * 10.0);
                QColor color = m_brush.color();
                color.setAlphaF(color.alphaF() * 0.25);
                grad.setColorAt(0, m_brush.color());
                grad.setColorAt(0.5, Qt::transparent);
                painter.setBrush(grad);
                qreal radius = grad.radius();
                painter.drawEllipse(event->position(), radius, radius);
                update(QRect(event->position().toPoint() - QPoint(radius, radius), QSize(radius * 2, radius * 2)));
            }
            break;

        default:
            {
                const QString error(tr("Unknown tablet device - treating as stylus"));
                qWarning() << error;
            }
            Q_FALLTHROUGH();
        case QInputDevice::DeviceType::Stylus: // e.g. one by wacom
            bool canRotate = event->pointingDevice()->capabilities().testFlag(QPointingDevice::Capability::Rotation);
            // one by wacom reports rotation capability but does not report rotation?
            // if device is 'one by wacom' {
            canRotate = false;
            // }
            if(canRotate) {
                m_brush.setStyle(Qt::SolidPattern);
                painter.setPen(Qt::NoPen);
                painter.setBrush(m_brush);
                QPolygonF poly;
                qreal halfWidth = pressureToWidth(mLastPoint.pressure);
                QPointF brushAdjust(qSin(qDegreesToRadians(-mLastPoint.rotation)) * halfWidth,
                                    qCos(qDegreesToRadians(-mLastPoint.rotation)) * halfWidth);
                poly << mLastPoint.pos + brushAdjust;
                poly << mLastPoint.pos - brushAdjust;
                halfWidth = m_pen.widthF();
                brushAdjust = QPointF(qSin(qDegreesToRadians(-event->rotation())) * halfWidth,
                                      qCos(qDegreesToRadians(-event->rotation())) * halfWidth);
                poly << event->position() - brushAdjust;
                poly << event->position() + brushAdjust;
                painter.drawConvexPolygon(poly);
                update(poly.boundingRect().toRect());
            } else {
                qreal maxPenRadius = pressureToWidth(1.0); // do we need this???
                painter.setPen(m_pen);
                painter.drawLine(mLastPoint.pos, event->position());
                update(QRect(mLastPoint.pos.toPoint(), event->position().toPoint()).normalized()
                       .adjusted(-maxPenRadius, -maxPenRadius, maxPenRadius, maxPenRadius));
            }
            break;
    }
}

void TabletCanvas::paintPixmap(QPainter &painter, QMouseEvent *event)
{

    qreal maxPenRadius = pressureToWidth(1.0); // do we need this???
    painter.setPen(m_pen);
    painter.drawLine(mLastPoint.pos, event->position());
    update(QRect(mLastPoint.pos.toPoint(), event->position().toPoint()).normalized()
           .adjusted(-maxPenRadius, -maxPenRadius, maxPenRadius, maxPenRadius));
    // painter.setPen(QPen(Qt::black, 5));  // Adjust the pen width/color as needed
    // painter.drawLine(mLastPoint.pos, event->position());
}
//! [5]

qreal TabletCanvas::pressureToWidth(qreal pressure)
{
    qreal pressureSens = penSize;
    return pressure * pressureSens + penSize;
}

//! [7]
void TabletCanvas::updateBrush(const QTabletEvent *event)
{
    int hue, saturation, value, alpha;
    m_color.getHsv(&hue, &saturation, &value, &alpha);

    int vValue = int(((event->yTilt() + 60.0) / 120.0) * 255);
    int hValue = int(((event->xTilt() + 60.0) / 120.0) * 255);
//! [7] //! [8]

    switch (m_alphaChannelValuator) {
        case PressureValuator:
            m_color.setAlphaF(event->pressure());
            break;
        case TangentialPressureValuator:
            if (event->deviceType() == QInputDevice::DeviceType::Airbrush)
                m_color.setAlphaF(qMax(0.01, (event->tangentialPressure() + 1.0) / 2.0));
            else
                m_color.setAlpha(255);
            break;
        case TiltValuator:
            m_color.setAlpha(std::max(std::abs(vValue - 127),
                                      std::abs(hValue - 127)));
            break;
        default:
            m_color.setAlpha(255);
    }

//! [8] //! [9]
    switch (m_colorSaturationValuator) {
        case VTiltValuator:
            m_color.setHsv(hue, vValue, value, alpha);
            break;
        case HTiltValuator:
            m_color.setHsv(hue, hValue, value, alpha);
            break;
        case PressureValuator:
            m_color.setHsv(hue, int(event->pressure() * 255.0), value, alpha);
            break;
        default:
            ;
    }

    switch (m_lineWidthValuator) {
        case PressureValuator:
            m_pen.setWidthF(pressureToWidth(event->pressure()));
            break;
        case TiltValuator:
            m_pen.setWidthF(std::max(std::abs(vValue - 127),
                                     std::abs(hValue - 127)) / 12);
            break;
        default:
            m_pen.setWidthF(penSize);
    }

    if (event->pointerType() == QPointingDevice::PointerType::Eraser) {
        m_brush.setColor(Qt::white);
        m_pen.setColor(Qt::white);
        m_pen.setWidthF(event->pressure() * 10 + penSize);
    } else {
        m_brush.setColor(m_color);
        m_pen.setColor(m_color);
    }
}

void TabletCanvas::updateBrush(const QMouseEvent *event)
{
    // stay with default settings?

    // m_brush.setColor(Qt::black);
    // m_pen.setColor(Qt::black);
    // m_pen.setWidthF( penSize);
}


void TabletCanvas::resizeEvent(QResizeEvent *)
{
    initPixmap();
}
