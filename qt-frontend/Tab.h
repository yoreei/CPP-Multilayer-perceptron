
// Copyright (C) 2016 The Qt Company Ltd.
// SPDX-License-Identifier: LicenseRef-Qt-Commercial OR BSD-3-Clause

#ifndef TABDIALOG_H
#define TABDIALOG_H

#include <QDialog>

QT_BEGIN_NAMESPACE
class QDialogButtonBox;
class QFileInfo;
class QTabWidget;
QT_END_NAMESPACE

//! [0]
class DrawPredictTab : public QWidget
{
    Q_OBJECT

public:
    explicit DrawPredictTab(QWidget *parent = nullptr);
};
//! [0]


//! [1]
class BrowseBadTab : public QWidget
{
    Q_OBJECT

public:
    explicit BrowseBadTab(QWidget *parent = nullptr);
};

class Tab : public QDialog
{
    Q_OBJECT

public:
    explicit Tab(QWidget *parent = nullptr);

private:
    QTabWidget *tabWidget;
    QDialogButtonBox *buttonBox;
};
//! [3]

#endif
