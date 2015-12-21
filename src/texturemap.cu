// Environment map background

#include <optix.h>
#include <optixu/optixu_math_namespace.h>

using namespace optix;

struct PerRayData_radiance
{
  float3 result;
  float importance;
  int depth;
};

rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );
rtDeclareVariable(PerRayData_radiance, prd_radiance, rtPayload, );
rtDeclareVariable(float3, texcoord, attribute texcoord, );
rtTextureSampler<float4, 2>      diffuse_map; // Corresponds to OBJ mtl params

RT_PROGRAM void closest_hit_radiance()
{
  //const float3 uv = texcoord;
  float3 Kd = make_float3( tex2D( diffuse_map, texcoord.x, texcoord.y ) );
  prd_radiance.result = Kd;//make_float3( tex2D(tex_map, uv.x, uv.y) );
  //rtPrintf( "Environment texture color: %d, %d, %d!\n", prd_radiance.result.x, prd_radiance.result.y, prd_radiance.result.z );
}



/*#include <optix.h>
#include <optixu/optixu_math_namespace.h>

using namespace optix;

// surface textures
rtTextureSampler<float, 2>     kd_map;

struct PerRayData_tex
{
  float3 result;
};

rtDeclareVariable(PerRayData_tex, prd, rtPayload, );
rtDeclareVariable(float3, texcoord, attribute texcoord, );




RT_PROGRAM void closest_hit_radiance()
{
  const float3 uv = texcoord;

  prd.result = make_float3(1.0f, 1.0f, 1.0f);//make_float3( tex2D( kd_map, uv.x, uv.y ) );
}
*/

