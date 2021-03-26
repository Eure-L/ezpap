#ifndef __CL_H
#define __CL_H

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define CL_TARGET_OPENCL_VERSION 120

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/opencl.h>
#endif

#endif /* __CL_H */
