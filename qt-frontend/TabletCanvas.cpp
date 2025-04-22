#include "TabletCanvas.h"
#include <QCoreApplication>
#include <QPainter>
#include <QShortcut>
#include <QtMath>

TabletCanvas::TabletCanvas(QPixmap* pixmapPtr, QWidget* parent)
    : QWidget(parent), m_brush(m_color)
    , m_pen(m_brush, 1, Qt::SolidLine, Qt::RoundCap, Qt::RoundJoin)
    , mPixmapPtr(pixmapPtr)
{

    setMouseTracking(true);
    setAttribute(Qt::WA_TabletTracking); // see docs for `tabletTracking`

    QShortcut* shortcut = new QShortcut(QKeySequence("Ctrl+S"), this);
    QShortcut* spaceShortcut = new QShortcut(QKeySequence(Qt::Key_Space), this);
    connect(shortcut, &QShortcut::activated, this, &TabletCanvas::saveImage);
    connect(spaceShortcut, &QShortcut::activated, this, &TabletCanvas::clear);

    initPixmap();
}

bool TabletCanvas::saveImage()
{
    return mPixmapPtr->scaled(28, 28).save("C:\\DATA\\git\\cpp-mlp\\assets.ignored\\tabletOut.png");
}

void TabletCanvas::clear()
{
    mPixmapPtr->fill(Qt::black);
    update();
}

void TabletCanvas::tabletEvent(QTabletEvent *event)
{
    updateBrush(event);
    mCurrentPoint.pos = mapTo28(event->position());
    mCurrentPoint.pressure = event->pressure();
    mCurrentPoint.rotation = event->rotation();

    switch (event->type()) {
    case QEvent::TabletPress:
        mDrawing = true;
        break;
    case QEvent::TabletMove:
//#ifndef Q_OS_IOS // not interestedin iOS at this point
        //if (event->pointingDevice() && event->pointingDevice()->capabilities().testFlag(QPointingDevice::Capability::Rotation))
        //updateCursor(event);
//#endif
        if (mDrawing) {
            updateBrush(event);
            m_pen.setColor(m_color);
            QPainter painter(mPixmapPtr);
            paintPixmap(painter);
            // emit bitmapUpdated(mPixmapPtr.get()); // possible optimization:
            // right now, worker thread crunches constantly, we could give it a rest with signals
        }
        break;
        case QEvent::TabletRelease:
            mDrawing = false;
            break;
        default:
            break;
        }


    mLastPoint.pos      = mCurrentPoint.pos;
    mLastPoint.pressure = mCurrentPoint.pressure;
    mLastPoint.rotation = mCurrentPoint.rotation;

    update();
    event->accept();
}

QPointF TabletCanvas::mapTo28(const QPointF& widgetPos) {
    auto w = width();
    auto h = height();
    assert(width() == height());
    float ratio = 28.f / width();
    return widgetPos * ratio;
}

void TabletCanvas::mouseMoveEvent(QMouseEvent *event)
{
    mCurrentPoint.pos = mapTo28(event->position());
    mCurrentPoint.pressure = 1;
    mCurrentPoint.rotation = 0;

    if (event->buttons() & Qt::LeftButton) {
        updateBrush(event);

        QPainter painter(mPixmapPtr);
        paintPixmap(painter);
    }

    mLastPoint.pos      = mCurrentPoint.pos;
    mLastPoint.pressure = 1;
    mLastPoint.rotation = 0;

    update();
    event->accept();
}

void TabletCanvas::initPixmap()
{
    *mPixmapPtr = QPixmap(28, 28);
    // this is for high-resolution displays and smooth visuals:
    // qreal dpr = devicePixelRatio();
    // mPixmapPtr->setDevicePixelRatio(dpr);
    clear();
}

void TabletCanvas::paintEvent(QPaintEvent *event)
{
    assert(mPixmapPtr);
    QPainter painter(this);

    QPixmap pixelatedPixmap = mPixmapPtr->scaled(
        size(),
        Qt::KeepAspectRatio,
        Qt::FastTransformation
        );
    painter.drawPixmap(0,0, pixelatedPixmap);

    QImage mImage = mPixmapPtr->scaled(28, 28).toImage().convertToFormat(QImage::Format_Grayscale8);
    mImage.invertPixels();
}

void TabletCanvas::paintPixmap(QPainter &painter)
{
    painter.setRenderHint(QPainter::Antialiasing);
    qreal maxPenRadius = pressureToWidth(1.0);
    painter.setPen(m_pen);
    painter.drawLine(mLastPoint.pos, mCurrentPoint.pos);
    update(QRect(mLastPoint.pos.toPoint(), mCurrentPoint.pos.toPoint()).normalized()
           .adjusted(-maxPenRadius, -maxPenRadius, maxPenRadius, maxPenRadius));
}

/// pressure is from 0 to 1
qreal TabletCanvas::pressureToWidth(qreal pressure)
{
    return penSize + (pressure - .8f) * penSize * 0.5;
}

void TabletCanvas::updateBrush(const QTabletEvent *event)
{
    const auto& capabilities = event->pointingDevice()->capabilities();

    if (capabilities.testFlag(QPointingDevice::Capability::Rotation)) {
        mCurrentPoint.rotation = event->rotation();
    }
    else {
        mCurrentPoint.rotation = 0;
    }

    if (capabilities.testFlag(QPointingDevice::Capability::Pressure)) {
        m_pen.setWidthF(pressureToWidth(event->pressure()));
    }
    else {
        m_pen.setWidthF(pressureToWidth(1));
    }

    m_brush.setColor(m_color);
    m_pen.setColor(m_color);
}

void TabletCanvas::updateBrush(const QMouseEvent *event)
{
    m_brush.setColor(Qt::white);
    m_pen.setColor(Qt::white);
    m_pen.setWidthF(penSize);
}


