//
//  Sample5Scene.h
//  Optix
//
//  Created by Hao Luo on 3/13/15.
//  Copyright (c) 2015 Hao Luo. All rights reserved.
//

#ifndef SMAPLE5_H
#define SMAPLE5_H

//#include <optixu/optixpp_namespace.h>
#include "glwidget.h"
#include "MeshScene.h"
#include "OptixMesh.h"

#include <QImage>
#include <QKeyEvent>

using namespace optix;

class Sample5Scene : public MeshScene
{
public:
    //~Sample5Scene();
    //
    // Helper types
    //
    enum ShadeMode
    {
      SM_PHONG=0,
      SM_AO,
      SM_NORMAL,
      SM_ONE_BOUNCE_DIFFUSE,
      SM_AO_PHONG
    };

    enum CameraMode
    {
      CM_PINHOLE=0,
      CM_ORTHO
    };

    //
    // Sample5Scene specific
    //
    Sample5Scene();

    // Setters for controlling application behavior
    void setShadeMode( ShadeMode mode )              { m_shade_mode = mode;               }
    void setCameraMode( CameraMode mode )            { m_camera_mode = mode;              }
    void setAORadius( float ao_radius )              { m_ao_radius = ao_radius;           }
    void setAOSampleMultiplier( int ao_sample_mult ) { m_ao_sample_mult = ao_sample_mult; }
    void setLightScale( float light_scale )          { m_light_scale = light_scale;       }
    void setAA( bool onoff )                         { m_aa_enabled = onoff;              }
    void setGroundPlane( bool onoff )                { m_ground_plane_enabled = onoff;    }
    void setAnimation( bool anim )                   { m_animation = anim;                }

    //From SampleScene
    void initScene( InitialCameraData& camera_data );
    void trace( const RayGenCameraData& camera_date );
    //void doResize( unsigned int width, unsigned height );
    optix::Buffer getOutputBuffer();
    bool keyPressEvent( int key );

    void setAdaptiveAA( bool adaptive_aa ) { m_adaptive_aa = adaptive_aa; }
    bool adaptive_aa(){ return m_adaptive_aa; }

private:
    int getEntryPoint() { return m_adaptive_aa ? AdaptivePinhole: Pinhole; }
    enum {
        Pinhole = 0,
        AdaptivePinhole = 1
    };

    void initContext();
    void initLights();
    void initMaterial();
    void InsertModel(const std::string& name, GeometryGroup ggroup, Material mat, Program mint, bool suppressErrors);
    void initGeometry();
    void initCamera( InitialCameraData& cam_data );
    void preprocess();
    void resetAccumulation();
    void genRndSeeds( unsigned int width, unsigned int height );

    // Helper functions
    void makeMaterialPrograms( Material material, const char *filename,
                                                  const char *ch_program_name,
                                                  const char *ah_program_name );


    CameraMode    m_camera_mode;

    ShadeMode     m_shade_mode;
    bool          m_aa_enabled;
    bool          m_ground_plane_enabled;
    float         m_ao_radius;
    int           m_ao_sample_mult;
    float         m_light_scale;

    Material      m_material, normal_matl, metal_matl, glass_matl, floor_matl, diffuse_matl;
    Material      organs_matl, rat_matl, eyes_matl, teeth_matl, water_matl,bones_matl, body_matl, ground_matl;
    MeshMaterialParams  m_material_params;
    Aabb          m_aabb;
    Buffer        m_rnd_seeds;
    Buffer        m_accum_buffer;
    bool          m_accum_enabled;

    float         m_scene_epsilon;
    int           m_frame;
    bool          m_animation;

    //From sample5
    unsigned int  m_frame_number;
    bool          m_adaptive_aa;

    std::string   datapath( const std::string& base );
    std::string   texture_path;
};
#endif // SMAPLE5_H
