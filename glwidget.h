#ifndef GLWIDGET_H
#define GLWIDGET_H

#include <QGLWidget>

#include <optixu/optixpp_namespace.h>
#include "utility.h"
#include "SampleScene.h"

//#include "sample1.h"
//#include "sample2.h"



class Mouse;
class PinholeCamera;


//-----------------------------------------------------------------------------
//
// OpenGL display
//
//-----------------------------------------------------------------------------
class glWidget : public QGLWidget
{
    Q_OBJECT

public:
    explicit glWidget(QWidget* _parent = 0);
    ~glWidget();


    enum contDraw_E { CDNone=0, CDProgressive=1, CDAnimated=2, CDBenchmark=3, CDBenchmarkTimed=4 };

    static contDraw_E getContinuousMode() { return m_app_continuous_mode; }
    static void setContinuousMode(contDraw_E continuous_mode);
    static void restartProgressiveTimer();

protected:
    void resizeGL(int width, int height);
    void initializeGL();
    void paintGL();
    //void paintEvent(QPaintEvent *event);

    void getMayaCamera();

    void mouseMoveEvent( QMouseEvent *event );
    void mousePressEvent( QMouseEvent *event );
    void mouseReleaseEvent( QMouseEvent *event );

private:

    //Sample1* sample1Scene;
    //Sample2* sample2Scene;

    // Draw text to screen at window pos x,y.  To make this public we will need to have
    // a public helper that caches the text for use in the display func
    static void drawText( const std::string& text, float x, float y, void* font );
    static QPainter* painter;

    // Do the actual rendering to the display
    static void displayFrame();

    // Set the current continuous drawing mode, while preserving the app's choice.
    static void setCurContinuousMode(contDraw_E continuous_mode);

    // Cleans up the rendering context and quits.  If there wasn't error cleaning up, the
    // return code is passed out, otherwise 2 is used as the return code.
    static void quit(int return_code=0);


    static void display();

    static Mouse*         m_mouse;
    static PinholeCamera* m_camera;
    static SampleScene*   m_scene;

    static double         m_last_frame_time;
    static unsigned int   m_last_frame_count;
    static unsigned int   m_frame_count;

    static bool           m_display_fps;
    static double         m_fps_update_threshold;
    static char           m_fps_text[32];
    static optix::float3  m_text_color;
    static optix::float3  m_text_shadow_color;

    static bool           m_print_mem_usage;

    static contDraw_E     m_app_continuous_mode;
    static contDraw_E     m_cur_continuous_mode;

    static bool           m_display_frames;
    static bool           m_save_frames_to_file;
    static std::string    m_save_frames_basename;

    static unsigned int   m_texId;
    static bool           m_sRGB_supported;
    static bool           m_use_sRGB;

    static double         m_progressive_timeout; // how long to do continuous rendering for progressive refinement (ignored when benchmarking or animating)

    static int            m_num_devices;

    static bool           m_enable_cpu_rendering; // enables CPU execution of OptiX programs

public:
    static SampleScene* getScene(){ return m_scene; }

    static void printMemUsage( bool checked) { m_print_mem_usage = checked ;}
    static void displayFps( bool checked) { m_display_fps = checked ;}

    std::string exec(char* cmd);
    //----------------------------------------------------------------------------
    // Embeeding python interface
    //----------------------------------------------------------------------------
    //static void* python_run(void* thread_nr);
    //static void  multi_thread( void* func(void* thread_nr) );
};

#endif // GLWIDGET_H
