#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include "thread.h"

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();

    void keyPressEvent( QKeyEvent* event );


    // Pubnub connection implementation
    bool Pubnub_connected(){ return connected; }
    void set_QApplication( QApplication* a ){ qapplication = a; }

private slots:
    void on_Slider_sphere_size_valueChanged(int value);

    void on_Slider_ior_valueChanged(int value);

    void on_Slider_light_position_valueChanged(int value);

    void on_checkBox_memory_clicked(bool checked);

    void on_checkBox_fps_clicked(bool checked);

    void on_checkBox_acceleration_clicked(bool checked);

    void on_checkBox_antialiasing_clicked(bool checked);

    void on_Slider_paint_camera_valueChanged(int value);

    void on_comboBox_paint_camera_activated(int index);

    void on_toolButton_paint_camera_clicked();

    void on_checkBox_clicked(bool checked);

    void on_pushButton_Pubnub_clicked();

private:
    Ui::MainWindow *ui;

    QApplication* qapplication;
    bool connected = false;
    Thread* t;
};

#endif // MAINWINDOW_H
