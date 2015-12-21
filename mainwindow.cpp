#include "mainwindow.h"
#include "ui_mainwindow.h"

#include <QKeyEvent>
#include <QFileDialog>
#include <iostream>

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    t = new Thread();
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::keyPressEvent(QKeyEvent* event)
{
    bool checked;

    try {
      if( glWidget::getScene()->keyPressEvent( event->key() ) ){
        ui->widget->updateGL();
        //return;
      }
    } catch( optix::Exception& e ){
      sutilReportError( e.getErrorString().c_str() );
      exit(2);
    }

    switch(event->key()) {
    case Qt::Key_Escape:
        close();
        break;

    case Qt::Key_M:

        if( ui->checkBox_memory->isChecked() )
            checked = false;
        else
            checked = true;
        ui->checkBox_memory->setChecked( checked );
        ui->widget->printMemUsage( checked );
        ui->widget->updateGL();
        break;

    case Qt::Key_D:
        if( ui->checkBox_fps->isChecked() )
            checked = false;
        else
            checked = true;
        ui->checkBox_fps->setChecked( checked );
        ui->widget->displayFps( checked );
        ui->widget->updateGL();
        break;

    case Qt::Key_A:
        if( ui->checkBox_antialiasing->isChecked() )
            checked = false;
        else
            checked = true;
        ui->checkBox_antialiasing->setChecked( checked );

    default:
        event->ignore();
        break;
    }
}

void MainWindow::on_Slider_sphere_size_valueChanged(int value)
{
    float radius = value / 10.0f;
    ui->widget->getScene()->updateSphere( radius );
    ui->widget->updateGL();
}

void MainWindow::on_Slider_ior_valueChanged(int value)
{
    float ior = value / 10.0f;
    ui->widget->getScene()->updateMaterial( ior );
    ui->widget->updateGL();
}

void MainWindow::on_Slider_light_position_valueChanged(int value)
{
    ui->widget->getScene()->updateLights( 0, value );
    ui->widget->updateGL();
}

void MainWindow::on_checkBox_memory_clicked(bool checked)
{
    glWidget::printMemUsage( checked );
    ui->widget->updateGL();
}

void MainWindow::on_checkBox_fps_clicked(bool checked)
{
    glWidget::displayFps( checked );
    ui->widget->updateGL();
}

void MainWindow::on_checkBox_acceleration_clicked(bool checked)
{
    glWidget::getScene()->updateAcceleration( checked );
    ui->widget->updateGL();
}

void MainWindow::on_checkBox_antialiasing_clicked(bool checked)
{
    glWidget::getScene()->keyPressEvent( Qt::Key_A );
    ui->widget->updateGL();
}

void MainWindow::on_Slider_paint_camera_valueChanged(int value)
{
    float scale = value / 10.0f;
    glWidget::getScene()->updatePaintCamera( scale );
    ui->widget->updateGL();
}

void MainWindow::on_comboBox_paint_camera_activated(int index)
{
    glWidget::getScene()->paintCameraType( index );
    ui->widget->updateGL();
}

void MainWindow::on_toolButton_paint_camera_clicked()
{
    QString fileName;
    fileName = QFileDialog::getOpenFileName(this,
                                            tr("Load Image"),
                                            "/Users/haoluo/qt-workspace",
                        tr("Image Files (*.png *.jpg *.bmp *.ppm *.hdr)"));
    glWidget::getScene()->paintCameraImage( fileName.toStdString() );
    ui->widget->updateGL();
}

void MainWindow::on_checkBox_clicked(bool checked)
{
    glWidget::getScene()->updateEnvmapOnOff( checked );
    ui->widget->updateGL();
}

void MainWindow::on_pushButton_Pubnub_clicked()
{
    connected = !connected;
    qDebug()<<"Connection: "<<connected;
    //printf("Connection: %i \n", connected);

    if(connected ){
        //QObject::connect(t, SIGNAL(finished()), qapplication, SLOT(quit()) );
        t->start();
    }else{
        t->quit();
    }
}
