#ifndef UTILITY_H
#define UTILITY_H

#include <optix.h>

//class Utility
//{
//public:
//    Utility();
//    ~Utility();
//};





#ifdef __cplusplus
extern "C" {
#endif

/************************************
 **
 **    Error checking helpers
 **
 ***********************************/

void sutilReportError(const char* message);


/************************************
 **
 **    Display helper functions
 **
 ***********************************/

/* Must be called before sutilDisplay*InGlutWindow().  This function
 * may only be called once per application invocation.  Normally this
 * should be called before your application's parsing of command line
 * args since glut will modify the given argument list by removing glut-
 * specific arguments.
 */

/* Get current time in seconds for benchmarking/timing purposes.
 *   current_time      : return param for the current time in double
 */
RTresult  sutilCurrentTime( double* current_time );


#ifdef __cplusplus
}
#endif

#endif // UTILITY_H
