#include "kernel/ocl/common.cl"
typedef char cell_t;

#define R 255
#define G 255
#define B 0
#define bits sizeof(cell_t)*8
#define AVXBITS 256
#define VEC_SIZE (AVXBITS/(sizeof(cell_t)*8))
#define NB_FAKE_X ((DIM/GPU_TILE_W)+2)
#define NB_FAKE_Y ((DIM/GPU_TILE_H)+2)

////////////////    Tools
static inline int table_cell ( int y, int x)
{
    return ((y)* (DIM) + x); 
}

static inline int table_cell_hybrid ( int y, int x)
{
    return (y+1) * (DIM+VEC_SIZE*2) + (x + VEC_SIZE); 
}
#define table(y, x) (table_cell ((y), (x)))
#define table_h(y, x) (table_cell_hybrid ((y), (x)))



// //////////Bitmap
static inline int table_map ( int x, int y)
{
  return  (y+1) * NB_FAKE_X + (x+1);
}
#define map(x, y) (table_map ( (x), (y)))


// static inline unsigned getBitCellRow(cell_t cell, unsigned i){
//   return cell>>((i)%bits)&(0x01);
// }

static inline bool hasNeighbourChanged(unsigned i,unsigned j, __global char * mapp){

  return  *(mapp+map(i,j))|*(mapp+map(i-1,j))|*(mapp+map(i+1,j))|*(mapp+map(i,j-1))|*(mapp+map(i,j+1))
  |*(mapp+map(i-1,j-1))|*(mapp+map(i+1,j+1))|*(mapp+map(i+1,j-1))|*(mapp+map(i-1,j+1));
}

//////////////////  OCL VERSIONS
__kernel void life_ocl(__global cell_t * in, __global cell_t * out,__global bool * change){

    int y = get_global_id (1);
    int x = get_global_id (0);
    

    if(x>0 && y> 0 && x< DIM && y<DIM){ 
        unsigned  n = 0;
        unsigned me  = in[table_h(y,x)] != 0;

        n += in[table_h(y,x)];
        n += in[table_h(y-1,x)];
        n += in[table_h(y,x-1)];
        n += in[table_h(y-1,x-1)];
        n += in[table_h(y+1,x)];
        n += in[table_h(y,x+1)];
        n += in[table_h(y+1,x+1)];
        n += in[table_h(y+1,x-1)];
        n += in[table_h(y-1,x+1)];    

        n = (n == 3 + me) | (n == 3);
        
        if (n != me){
            *change = 1;
        }
        
        out[table_h(y,x)]=n;
    }
    
}

__kernel void life_ocl_hybrid(__global cell_t * in, __global cell_t * out,__global bool * change,
                                unsigned  cpu_y_part){
    int y = get_global_id (1) + cpu_y_part;
    int x = get_global_id (0);
    
    if(x>0 && y> 0 && x< DIM && y<DIM){ 
        cell_t  n = 0;
        cell_t me  = in[table_h(y,x)] != 0;

        n += in[table_h(y,x)];
        n += in[table_h(y-1,x)];
        n += in[table_h(y,x-1)];
        n += in[table_h(y-1,x-1)];
        n += in[table_h(y+1,x)];
        n += in[table_h(y,x+1)];
        n += in[table_h(y+1,x+1)];
        n += in[table_h(y+1,x-1)];
        n += in[table_h(y-1,x+1)];    

        n = (n == 3 + me) | (n == 3);
        
        if (n != me)
            *change = 1;
        
        out[table_h(y,x)]=n;
    }
}

__kernel void life_ocl_bits(__global cell_t * in, __global cell_t * out,__global bool * change){

    int y = get_global_id (1);
    int x = get_global_id (0)*bits;

    // the thread operates on the word containing the desired cell

    out[y*DIM+x] = 1;
         
}

#if 1

// GPU-only version

__kernel void life_update_texture (__global cell_t *cur, __write_only image2d_t tex)
{   
    int y = get_global_id (1);
    int x = get_global_id (0);
    int2 pos = (int2)(x, y);
    
    unsigned color = cur [(y+1) * (DIM+VEC_SIZE*2) + (x + VEC_SIZE)];

    write_imagef (tex, (int2)(x, y), color_scatter (color*0xFFFF00FF));




}

#else

// hybrid version

__kernel void life_update_texture (__global cell_t *cur, __write_only image2d_t tex)
{
    int y = get_global_id (1);
    int x = get_global_id (0);
    int2 pos = (int2)(x, y);
    
    unsigned color = cur [(y+1) * (DIM+VEC_SIZE*2) + (x + VEC_SIZE)];

    write_imagef (tex, (int2)(x, y), color_scatter (color*0xFFFF00FF));

}

// #else
// __kernel void life_update_texture (__global cell_t *cur, __write_only image2d_t tex)
// {
//     int y = get_global_id (1);
//     int x = get_global_id (0);
//     int2 pos = (int2)(x, y);
//     unsigned c = ((cur[y * DIM + x/bits])>> x%bits) & 0x01;
//     printf("%d", cur[y * DIM + x/bits]);

//     c = rgba(R*c, G*c, B*c, 0xFF);

//     write_imagef (tex, pos, color_scatter (c));
// }

#endif