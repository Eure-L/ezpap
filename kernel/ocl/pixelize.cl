#include "kernel/ocl/common.cl"

#ifdef PARAM
#define PIX_BLOC PARAM
#else
#define PIX_BLOC 16
#endif

// In this over-simplified kernel, all the pixels of a bloc adopt the color
// on the top-left pixel (i.e. we do not compute the average color).
__kernel void pixelize_ocl_2 (__global unsigned *in)
{
  __local unsigned couleur [GPU_TILE_H / PIX_BLOC][GPU_TILE_W / PIX_BLOC];
  int x = get_global_id (0);
  int y = get_global_id (1);
  int xloc = get_local_id (0);
  int yloc = get_local_id (1);

  if (xloc % PIX_BLOC == 0 && yloc % PIX_BLOC == 0)
    couleur [yloc / PIX_BLOC][xloc / PIX_BLOC] = in [y * DIM + x];
  
  
  barrier (CLK_LOCAL_MEM_FENCE);

  in [y * DIM + x] = couleur [yloc / PIX_BLOC][xloc / PIX_BLOC];
}


__kernel void pixelize_ocl (__global unsigned *in)
{
  __local int4 couleur [GPU_TILE_H][GPU_TILE_W];

  int x = get_global_id (0);
  int y = get_global_id (1);
  int xloc = get_local_id (0);
  int yloc = get_local_id (1);


  couleur [yloc][xloc] = color_to_int4(in [(y) * DIM + x]);
  
  barrier (CLK_LOCAL_MEM_FENCE);
  
  //reduction en x sur toutes les lignes
  for(int i=GPU_TILE_W/2; i>0; i= i/2){
    barrier (CLK_LOCAL_MEM_FENCE);
    if(xloc < i)
      couleur [yloc][xloc] += couleur [yloc][xloc+i]; 
  }
  //reduction en y sur la premiere colonne
  for(int j=GPU_TILE_H/2; j>0; j= j/2){
    barrier (CLK_LOCAL_MEM_FENCE);
    if(yloc < j && xloc == 0)
      couleur [yloc][xloc] += couleur [yloc+j][xloc];
  }
  
  barrier (CLK_LOCAL_MEM_FENCE);
  in [y * DIM + x] =  int4_to_color(couleur [0][0] /(int4)(GPU_TILE_H * GPU_TILE_W));
}
