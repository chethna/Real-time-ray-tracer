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

rtTextureSampler<float4, 2> envmap;

RT_PROGRAM void envmap_miss()
{
  float theta = atan2f( ray.direction.x, ray.direction.z );
  float phi   = M_PI_2f +  asinf( ray.direction.y );
  float u     = (theta + M_PIf) * (0.5f * M_1_PIf);
  float v     =  M_1_PIf * (phi);
  prd_radiance.result = make_float3( tex2D(envmap, u, v) );
  //rtPrintf( "Environment texture color: %d, %d, %d!\n", prd_radiance.result.x, prd_radiance.result.y, prd_radiance.result.z );
}
