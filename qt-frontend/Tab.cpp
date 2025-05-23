#include <QtWidgets>

#include "Tab.h"
#include "BrowseMnist.h"
#include "DrawPredict.h"


void dumpWidgetTree(QWidget* w, int depth = 0)
{
    QString indent(depth*2, ' ');
    qDebug() << indent
             << w->metaObject()->className()
             << w->objectName()
             << w->geometry()
             << w->sizePolicy()
             << "hint: " << w->sizeHint();
    for (QObject* c : w->children()) {
        if (QWidget* cw = qobject_cast<QWidget*>(c))
            dumpWidgetTree(cw, depth+1);
    }
}

Tab::Tab(QWidget *parent)
    : QDialog(parent)
{
    tabWidget = new QTabWidget;

    // Size of Tab widget is determined by the child widgets
    setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Minimum);

    tabWidget->addTab(new DrawPredictTab(this),tr("Live Predict"));
    tabWidget->addTab(new BrowseBadTab(this), tr("Browse Mnist"));
    QVBoxLayout *mainLayout = new QVBoxLayout;
    mainLayout->addWidget(tabWidget);
    setLayout(mainLayout);
    setWindowTitle(tr("Danny's MLP with C++"));

    QShortcut* enterShortcut = new QShortcut(QKeySequence("Ctrl+A"), this);
    connect(enterShortcut, &QShortcut::activated, this, [this](){dumpWidgetTree(this);});
}

DrawPredictTab::DrawPredictTab(QWidget *parent)
    : QWidget(parent)
{
    DrawPredict* drawPredict = new DrawPredict();

    QVBoxLayout *layout= new QVBoxLayout;
    layout->addWidget(drawPredict);
    setLayout(layout);
}

BrowseBadTab::BrowseBadTab(QWidget *parent)
    : QWidget(parent)
{
    BrowseBad *browseBad = new BrowseBad();

    QVBoxLayout *layout = new QVBoxLayout;
    layout->addWidget(browseBad);
    setLayout(layout);
}
