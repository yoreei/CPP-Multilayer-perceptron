// Copyright (C) 2016 The Qt Company Ltd.
// SPDX-License-Identifier: LicenseRef-Qt-Commercial OR BSD-3-Clause

#include <QtWidgets>

#include "Tab.h"
#include "BrowseBad.h"
#include "drawpredict.h"

//! [0]
Tab::Tab(QWidget *parent)
    : QDialog(parent)
{
    tabWidget = new QTabWidget;
    tabWidget->addTab(new DrawPredictTab(),tr("🖌️Draw Predict"));
    tabWidget->addTab(new BrowseBadTab(), tr("👎Browse Bad"));
//! [0]

//! [1] //! [2]
    buttonBox = new QDialogButtonBox(QDialogButtonBox::Ok
//! [1] //! [3]
                                     | QDialogButtonBox::Cancel);

    connect(buttonBox, &QDialogButtonBox::accepted, this, &QDialog::accept);
    connect(buttonBox, &QDialogButtonBox::rejected, this, &QDialog::reject);
//! [2] //! [3]

//! [4]
    QVBoxLayout *mainLayout = new QVBoxLayout;
    mainLayout->addWidget(tabWidget);
    mainLayout->addWidget(buttonBox);
    setLayout(mainLayout);
//! [4]

//! [5]
    setWindowTitle(tr("Tab Dialog"));
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
