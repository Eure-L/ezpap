#include "kernel/ocl/common.cl"


__kernel void rotation90_ocl (__global unsigned *in, __global unsigned *out)
{
  int x = get_global_id (0);
  int y = get_global_id (1);


  out [(DIM - x - 1) * DIM + y] = in [y * DIM + x];
}

__kernel void rotation90_ocl_v2 (__global unsigned *in, __global unsigned *out)
{
  int x = get_global_id (0);
  int y = get_global_id (1);

  int xloc = get_local_id (0);
  int yloc = get_local_id (1);

  unsigned tampon[DIM][DIM];

  tampon [GPU_TILE_W-xloc-1][yloc] = in [y * DIM + x];

  out [DIM - (x-xloc)-GPU_TILE_W] = tampon [GPU_TILE_W-xloc-1][yloc];

}