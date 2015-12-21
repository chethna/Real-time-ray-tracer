#if defined(__APPLE__)
#  include <OpenGL/gl.h>
#else
#  include <GL/glew.h>
#  if defined(_WIN32)
#    include <GL/wglew.h>
#  endif
#  include <GL/gl.h>
#endif

#include "SampleScene.h"
#include <ImageLoader.h>

#include <optixu/optixu_math_namespace.h>
#include <optixu/optixu.h>

#include <iostream>
//#include <sstream>
//#include <cstdlib>
//#include <algorithm>

#include "commonStructs.h"

#include <mach-o/dyld.h>
#include <stdio.h>  /* defines FILENAME_MAX */

#ifdef WINDOWS
    #include <direct.h>
    #define GetCurrentDir _getcwd
#else
    #include <unistd.h>
    #define GetCurrentDir getcwd
#endif

using namespace optix;

//-----------------------------------------------------------------------------
//
// SampleScene class implementation
//
//-----------------------------------------------------------------------------

SampleScene::SampleScene()
  : m_camera_changed( true ), m_use_vbo_buffer( true ), m_num_devices( 0 ), m_cpu_rendering_enabled( true )
{
  m_context = Context::create();
  updateCPUMode();
}

//SampleScene::InitialCameraData::InitialCameraData( const std::string &camstr)
//{
//  std::istringstream istr(camstr);
//  istr >> eye >> lookat >> up >> vfov;
//}

const char* const SampleScene::ptxpath( const std::string& target, const std::string& base )
{
    static std::string path;
    char execPath[FILENAME_MAX];
    uint32_t size = sizeof(execPath);

    if( _NSGetExecutablePath(execPath, &size) != 0 )
        printf("buffer too small; need size %u\n", size );
    const std::string& dir = execPath;
    const std::string& d = dir.substr(0, dir.size()- sizeof("sample5")+1 );
    //std::cout<<"dir: "<<d<<"\n";

    //if (!GetCurrentDir(cCurrentPath, sizeof(cCurrentPath)))
    //    std::cerr<< "errno" <<std::endl;
    //cCurrentPath[sizeof(cCurrentPath) - 1] = '\0'; /* not really required */
    //printf ("The current working directory is %s \n", cCurrentPath);
    //const std::string& dir = cCurrentPath;

    path =  d + "ptx/" + target + "_generated_" + base + ".ptx";
    return path.c_str();
}

Buffer SampleScene::createOutputBuffer( RTformat format,
                                        unsigned int width,
                                        unsigned int height )
{
  // Set number of devices to be used
  // Default, 0, means not to specify them here, but let OptiX use its default behavior.
  if(m_num_devices)
  {
    int max_num_devices    = Context::getDeviceCount();
    int actual_num_devices = std::min( max_num_devices, std::max( 1, m_num_devices ) );
    std::vector<int> devs(actual_num_devices);
    for( int i = 0; i < actual_num_devices; ++i ) devs[i] = i;
    m_context->setDevices( devs.begin(), devs.end() );
  }

  Buffer buffer;

  if ( m_use_vbo_buffer && !m_cpu_rendering_enabled )
  {
    /*
      Allocate first the memory for the gl buffer, then attach it to OptiX.
    */
    GLuint vbo = 0;
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    size_t element_size;
    m_context->checkError(rtuGetSizeForRTformat(format, &element_size));
    glBufferData(GL_ARRAY_BUFFER, element_size * width * height, 0, GL_STREAM_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    buffer = m_context->createBufferFromGLBO(RT_BUFFER_OUTPUT, vbo);
    buffer->setFormat(format);
    buffer->setSize( width, height );
  }
  else {
    buffer = m_context->createBuffer( RT_BUFFER_OUTPUT, format, width, height);
  }

  return buffer;
}

void SampleScene::cleanUp()
{
  m_context->destroy();
  m_context = 0;
}

void
SampleScene::trace( const RayGenCameraData& camera_data, bool& display )
{
  trace(camera_data);
}

void SampleScene::resize(unsigned int width, unsigned int height)
{
  try {
    Buffer buffer = getOutputBuffer();
    buffer->setSize( width, height );

    if(m_use_vbo_buffer)
    {
      buffer->unregisterGLBuffer();
      glBindBuffer(GL_PIXEL_UNPACK_BUFFER, buffer->getGLBOId());
      glBufferData(GL_PIXEL_UNPACK_BUFFER, buffer->getElementSize() * width * height, 0, GL_STREAM_DRAW);
      glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
      buffer->registerGLBuffer();
    }

  } catch( Exception& e ){
    sutilReportError( e.getErrorString().c_str() );
    exit(2);
  }

  // Let the user resize any other buffers
  //doResize( width, height );
}

void
SampleScene::setNumDevices( int ndev )
{
  m_num_devices = ndev;

  if (m_cpu_rendering_enabled && m_num_devices > 0) {
    rtContextSetAttribute(m_context.get()->get(), RT_CONTEXT_ATTRIBUTE_CPU_NUM_THREADS, sizeof(m_num_devices), &m_num_devices);
  }
}

void
SampleScene::enableCPURendering(bool enable)
{
  // Is CPU mode already enabled
  std::vector<int> devices = m_context->getEnabledDevices();
  bool isCPUEnabled = false;
  for(std::vector<int>::const_iterator iter = devices.begin(); iter != devices.end(); ++iter)
  {
    if (m_context->getDeviceName(*iter) == "CPU") {
      isCPUEnabled = true;
      break;
    }
  }

  // Already in desired state, good-bye.
  if (isCPUEnabled == enable)
    return;

  if (enable)
  {
    // Turn on CPU mode

    int ordinal;
    for(ordinal = m_context->getDeviceCount()-1; ordinal >= 0; ordinal--)
    {
      if (m_context->getDeviceName(ordinal) == "CPU") {
        break;
      }
    }
    if (ordinal < 0)
      throw Exception("Attempting to enable CPU mode, but no CPU device found");
    m_context->setDevices(&ordinal, &ordinal+1);
  } else
  {
    // Turn off CPU mode

    // For now, simply grab the first device
    int ordinal = 0;
    m_context->setDevices(&ordinal, &ordinal+1);
  }

  // Check this here, in case we failed to make it into GPU mode.
  updateCPUMode();
}

void
SampleScene::incrementCPUThreads(int delta)
{
  int num_threads;
  RTresult code = rtContextGetAttribute(m_context.get()->get(), RT_CONTEXT_ATTRIBUTE_CPU_NUM_THREADS, sizeof(num_threads), &num_threads);
  m_context->checkError(code);
  num_threads += delta;
  if (num_threads <= 0)
    num_threads = 1;
  setNumDevices(num_threads);
}

void
SampleScene::updateCPUMode()
{
  m_cpu_rendering_enabled = m_context->getDeviceName(m_context->getEnabledDevices()[0]) == "CPU";
  if (m_cpu_rendering_enabled)
    m_use_vbo_buffer = false;
}

//-----------------------------------------------------------------------------
//
// Scene update
//
//-----------------------------------------------------------------------------
void SampleScene::updateSphere( float radius)//, float center )
{
    Group            top_level_group = m_context["top_object"]-> getGroup();
    Transform        transform       = top_level_group->getChild<Transform>( 0 );
    GeometryGroup    geometrygroup   = transform->getChild<GeometryGroup>();
    GeometryInstance instance        = geometrygroup->getChild( 0 );
    Geometry         geometry          = instance->getGeometry();

    // Sphere
    //geometry["sphere"]->setFloat( center, 0, 0, radius );

    // Sphere Shell
    geometry["radius2"]->setFloat( radius );        // Set Radius
    //geometry["center"]->setFloat( center, 0, 0 );   // set center

    // Mark Dirty
    int childCount = top_level_group->getChildCount();
    for ( int i = 0; i < childCount; i++) {
        if( top_level_group->getChildType( i ) == RT_OBJECTTYPE_TRANSFORM ){
            transform = top_level_group->getChild<Transform>( i );
            geometrygroup = transform->getChild<GeometryGroup>();
        }else{
            geometrygroup = top_level_group->getChild<GeometryGroup>( i );
            //geometrygroup = transform->getChild<GeometryGroup>();
        }
        geometrygroup->getAcceleration()->markDirty();
    }
    top_level_group->getAcceleration()->markDirty();
}

void SampleScene::updateMaterial( float refraction_index )
{
    Group            top_level_group = m_context["top_object"]-> getGroup();
    Transform        transform       = top_level_group->getChild<Transform>( 0 );
    GeometryGroup    geometrygroup   = transform->getChild<GeometryGroup>();
    GeometryInstance instance        = geometrygroup->getChild( 0 );
    Material         material        = instance->getMaterial( 0 );

    material["refraction_index"]->setFloat( refraction_index );
}

void SampleScene::updateLights(int index, float pos){
    index = 0;
    /* Lights */
    BasicLight lights[] = {
        { make_float3( pos, 40.0f, 30.0f ), make_float3( 1.0f, 1.0f, 1.0f ), 1 }
    };

    memcpy( m_context["lights"]->getBuffer()->map(), lights, sizeof(lights) );
    m_context["lights"]->getBuffer()->unmap();
}

void SampleScene::updateAcceleration( bool accel )
{
    if( accel )
        m_context["top_object"]->getGroup()->setAcceleration( m_context->createAcceleration( "Trbvh", "Bvh" ) );
    else
        m_context["top_object"]->getGroup()->setAcceleration( m_context->createAcceleration( "NoAccel", "NoAccel" ) );
}

void SampleScene::updatePaintCamera( float scale )
{
    m_context["paint_camera_scale"]->setFloat( scale );
}

void SampleScene::paintCameraType( unsigned int type )
{
    m_context["paint_camera_type"]->setUint( type );
}

void SampleScene::paintCameraImage( std::string fileName )
{
    const float3 default_color = make_float3(1.0f, 1.0f, 1.0f);
    int paint_type = m_context["paint_camera_type"]->getInt();

    if( paint_type == 0 ){
        m_context["camera_paint_map"]->setTextureSampler( loadTexture( m_context, fileName, default_color) );
    }
    else if ( paint_type == 1 )
        m_context["camera_pose_map"]->setTextureSampler( loadTexture( m_context, fileName, default_color) );
    else{
        m_context["camera_paint_map"]->setTextureSampler( loadTexture( m_context, fileName, default_color) );
        m_context["camera_pose_map"]->setTextureSampler( loadTexture( m_context, fileName, default_color) );
    }
}

std::string texpath( const std::string& base )
{
    std::string texture_path = "/Users/haoluo/qt-workspace/sample5/data";
    return texture_path + "/" + base;
}

void SampleScene::updateEnvmapOnOff(bool envmap)
{
    std::string ptx_path;
    if( envmap ){
        ptx_path = ptxpath( "sample5", "envmap.cu" );
        m_context->setMissProgram( 0, m_context->createProgramFromPTXFile( ptx_path, "envmap_miss" ) );
        const float3 default_color = make_float3(1.0f, 1.0f, 1.0f);
        m_context["envmap"]->setTextureSampler( loadTexture( m_context, texpath("autumn.ppm"), default_color) );
    }
    else{
        ptx_path = ptxpath( "sample5", "constantbg.cu" );
        m_context->setMissProgram( 0, m_context->createProgramFromPTXFile( ptx_path, "miss" ) );
        m_context["bg_color"]->setFloat( make_float3( 0.3f, 0.3f, 0.3f ) );
    }
}


void SampleScene::updateGeometry( float x, float y, float z )
{
    Group            top_level_group = m_context["top_object"]-> getGroup();
    Transform        transform       = top_level_group->getChild<Transform>( 1 );
    //GeometryGroup    geometrygroup   = m_geometry_group;
    //GeometryInstance instance        = geometrygroup->getChild( 0 );
    //Geometry         geometry          = instance->getGeometry();

    // Initial transform matrix
    const float tx=-2.0f, ty=0.0f, tz=0.0f;
    const float sx=x, sy=y, sz=z;
    // Matrices are row-major.
    float m[16] = { sx, 0, 0, tx,
                    0, sy, 0, ty,
                    0, 0, sz, tz,
                    0, 0, 0, 1 };

    //m[0] = m[5] = m[10] *= 5;

    transform->setMatrix(0, m, 0);
    top_level_group->getAcceleration()->markDirty();
}
