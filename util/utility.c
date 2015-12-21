#include "utility.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32)
#    define WIN32_LEAN_AND_MEAN
#    include<windows.h>
#    include<mmsystem.h>
#else /*Apple and Linux both use this */
#    include<sys/time.h>
#    include <unistd.h>
#    include <dirent.h>
#endif

//Utility::Utility()
//{

//}

//Utility::~Utility()
//{

//}

//void sutilReportError(const char* message)
//{
//  fprintf( stderr, "OptiX Error: %s\n", message );
//#if defined(_WIN32) && defined(RELEASE_PUBLIC)
//  {
//    char s[2048];
//    sprintf( s, "OptiX Error: %s", message );
//    MessageBox( 0, s, "OptiX Error", MB_OK|MB_ICONWARNING|MB_SYSTEMMODAL );
//  }
//#endif
//}

//#if defined(_WIN32)

//// inv_freq is 1 over the number of ticks per second.
//static double inv_freq;
//static int freq_initialized = 0;
//static int use_high_res_timer = 0;

//RTresult sutilCurrentTime( double* current_time )
//{
//  if(!freq_initialized) {
//    LARGE_INTEGER freq;
//    use_high_res_timer = QueryPerformanceFrequency(&freq);
//    inv_freq = 1.0/freq.QuadPart;
//    freq_initialized = 1;
//  }
//  if (use_high_res_timer) {
//    LARGE_INTEGER c_time;
//    if(QueryPerformanceCounter(&c_time)) {
//      *current_time = c_time.QuadPart*inv_freq;
//    } else {
//      return RT_ERROR_UNKNOWN;
//    }
//  } else {
//    *current_time = ( (double)timeGetTime() ) * 1.0e-3;
//  }
//  return RT_SUCCESS;
//}

//#else

//RTresult sutilCurrentTime( double* current_time )
//{
//  struct timeval tv;
//  if( gettimeofday( &tv, 0 ) ) {
//    fprintf( stderr, "sutilCurrentTime(): gettimeofday failed!\n" );
//    return RT_ERROR_UNKNOWN;
//  }

//  *current_time = tv.tv_sec+ tv.tv_usec * 1.0e-6;
//  return RT_SUCCESS;
//}

//#endif
