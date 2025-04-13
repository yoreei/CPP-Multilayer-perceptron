// Copyright (C) 2016 The Qt Company Ltd.
// SPDX-License-Identifier: LicenseRef-Qt-Commercial OR BSD-3-Clause

#include <QtWidgets>

#include "Tab.h"
#include "BrowseBad.h"
#include "DrawPredict.h"

//! [0]
Tab::Tab(QWidget *parent)
    : QDialog(parent)
{
    tabWidget = new QTabWidget;
    tabWidget->addTab(new DrawPredictTab(),tr("🖌️Draw Predict"));
    tabWidget->addTab(new BrowseBadTab(), tr("👎Browse Bad"));
    QVBoxLayout *mainLayout = new QVBoxLayout;
    mainLayout->addWidget(tabWidget);
    setLayout(mainLayout);
    setWindowTitle(tr("MLP with C++"));
}
//! [5]

//! [6]
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
