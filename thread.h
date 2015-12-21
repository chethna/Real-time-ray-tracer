#ifndef THREAD_H
#define THREAD_H

#include <QtCore>
#include <Python/Python.h>

class Thread : public QThread
{
public:
    void run()
    {
        //qDebug()<<"From worker thread: "<<currentThreadId();
        python_subscribe();
    }

    void quit()
    {
        globalDict = NULL;
    }

    void python_subscribe();
    void python_unsubscribe();

    static const char * const PubnubPath( const std::string& target);

    static void* python_retrieve_camera();

    static float     camera_array[];
    static bool      camera_array_changed;

    static float     geometry_array[];
    static bool      geometry_array_changed;

    static PyObject* globalDict;
    static PyObject* localDict;
};

#endif // THREAD_H






