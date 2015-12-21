#ifndef PAINT_CAMERA
#define PAINT_CAMERA

#include <optix_world.h>
#include <optixu/optixu_math_namespace.h>

rtTextureSampler<float4, 2> camera_paint_map;
rtTextureSampler<float4, 2> camera_pose_map;
rtDeclareVariable(float, paint_camera_scale, , ) = 0;

static __device__ inline float3 cameraTexture( float2 d, rtTextureSampler<float4, 2> map )
{
    //d /= 2.0f;
    return make_float3( tex2D(map, d.x, d.y) ) * paint_camera_scale;
//    float theta = atan2f( ray_direction.x, ray_direction.z );
//    float phi   = M_PIf * 0.5f -  acosf( ray_direction.y );
//    float u     = (theta + M_PIf) * (0.5f * M_1_PIf);
//    float v     = 0.5f * ( 1.0f + sin(phi) );

//    return make_float3( tex2D(map, u, v) ) * paint_camera_scale;
}

static __device__ inline float3 getColor( float3 ray_direction, rtTextureSampler<float4, 2> map )
{
    float theta = atan2f( ray_direction.x, ray_direction.z );
    float phi   = M_PIf * 0.5f -  acosf( ray_direction.y );
    float u     = (theta + M_PIf) * (0.5f * M_1_PIf);
    float v     = 0.5f * ( 1.0f + sin(phi) );

    return make_float3( tex2D(map, u, v) );
}

static __device__ inline void paint_camera( optix::Ray* ray )
{
    float3 color = getColor( ray->direction, camera_paint_map );

    //rtPrintf( "Camera texture color: %f, %f, %f!\n", map.x, map.y, map.z );

    // Apply painting on ray direction
    ray->direction = ray->direction + color * paint_camera_scale ;
}

static __device__ inline void pose_camera( optix::Ray* ray )
{
    float3 color = getColor( ray->direction, camera_pose_map );
    // Apply posing on ray origin
    ray->origin = ray->origin + color * paint_camera_scale * 2.0f ;
}

#endif // PAINT_CAMERA

