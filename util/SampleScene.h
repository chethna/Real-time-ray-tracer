#ifndef SAMPLESCENE_H
#define SAMPLESCENE_H

#include <optixu/optixpp_namespace.h>
#include "utility.h"

//-----------------------------------------------------------------------------
//
// SampleScene virtual class
//
//-----------------------------------------------------------------------------
class SampleScene
{
public:protected:
    typedef optix::float3 float3;
public:
    // Used to pass current camera info to the ray gen program at render time.
    // eye - Camera position
    // W   - Viewing direction.             length(W) -> focal distance
    // U   - Horizontal axis of view plane. length(U) -> view plane width at focal distance
    // V   - Vertical axis of view plane.   length(V) -> view plane height at focal distance
    struct RayGenCameraData
    {
      RayGenCameraData() {}
      RayGenCameraData( const float3& m_eye, const float3& m_U, const float3& m_V, const float3& m_W )
        : eye(m_eye), U(m_U), V(m_V), W(m_W) {}
      float3 eye;
      float3 U;
      float3 V;
      float3 W;
    };

    // Used to specify initial viewing parameters
    struct InitialCameraData
    {
      InitialCameraData() {}
      InitialCameraData( const std::string& camera_string );
      InitialCameraData( float3 m_eye, float3 m_lookat, float3 m_up, float  m_vfov )
        : eye(m_eye), lookat(m_lookat), up(m_up), vfov(m_vfov) {}

      float3 eye;
      float3 lookat;
      float3 up;
      float  vfov;
    };

    SampleScene();
    virtual ~SampleScene() {}

    void  signalCameraChanged() { m_camera_changed = true; }

    void  setNumDevices( int ndev );
    void  enableCPURendering(bool enable);
    void  incrementCPUThreads(int delta); // can pass in negative values

    void  setUseVBOBuffer( bool onoff ) { m_use_vbo_buffer = onoff; }
    bool  usesVBOBuffer() { return m_use_vbo_buffer; }

    static const char * const ptxpath( const std::string& target, const std::string& base );

    //----------------------------------------------------------------------------
    // Pure virtual interface to be implemented
    //----------------------------------------------------------------------------

    // Create the optix scene and return initial viewing parameters
    virtual void initScene( InitialCameraData &camera_data )=0;

    // Update camera shader with new viewing params and then trace
    virtual void trace( const RayGenCameraData &camera_data )=0;

    // Update camera shader with new viewing params and then trace
    virtual void trace( const RayGenCameraData& camera_data, bool& display );

    // Return the output buffer to be displayed
    virtual optix::Buffer getOutputBuffer()=0;

    //----------------------------------------------------------------------------
    // Optional virtual interface
    //----------------------------------------------------------------------------

    // This cleans up the Context.  If you override it, you should call
    // SampleScene::cleanUp() explicitly.
    virtual void   cleanUp();

    // Will resize the output buffer (which might use a VBO) then call doResize.
    // Override this if you want to handle ALL buffer resizes yourself.
    virtual void   resize(unsigned int width, unsigned int height);

    // Where derived classes should handle resizing all buffers except outputBuffer.
    //virtual void   doResize(unsigned int width, unsigned int height) {}

    // Use this to add additional keys. Some are already handled but
    // can be overridden.  Should return true if key was handled, false otherwise.
    virtual bool   keyPressed(unsigned char key, int x, int y) { return false; }
    virtual bool   keyPressEvent( int key ) { return false; }

    // Accessor
    optix::Context& getContext() { return m_context; }

    // Scene API
    void updateSphere( float radius );
    void updateGeometry(float x, float y, float z);
    void updateMaterial( float refraction_index );
    void updateLights( int index, float position );
    void updateAcceleration( bool accel );

    void updatePaintCamera( float scale );
    void paintCameraType( unsigned int type );
    void paintCameraImage( std::string fileName );
    void updateEnvmapOnOff( bool envmap );

protected:
    optix::Buffer createOutputBuffer(RTformat format, unsigned int width, unsigned int height);

    optix::Context m_context;

    bool   m_camera_changed;
    bool   m_use_vbo_buffer;
    int    m_num_devices;
    bool   m_cpu_rendering_enabled;

private:
  // Checks to see if CPU mode has been enabled and sets the appropriate flags.
  void updateCPUMode();
};

#endif // SAMPLESCENE_H
