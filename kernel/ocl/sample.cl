#include "kernel/ocl/common.cl"


__kernel void sample_ocl (__global unsigned *img)
{
  int x = get_global_id (0);
  int y = get_global_id (1);

  //unsigned color = 0xFFFF00FF; // opacity

  int red = x & 255;
  int blue = y & 255;

  img [y * DIM + x] = ((red << 24) | (blue <<8) | (255));
}
