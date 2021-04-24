#include "kernel/ocl/common.cl"


#define R 255
#define G 255
#define B 0

typedef unsigned cell_t;

static inline int table_cell ( int y, int x)
{
    return ((y)* (DIM) + x) * (sizeof(unsigned)/sizeof(cell_t)); 
}

#define table(y, x) (table_cell ((y), (x)))


__kernel void life_ocl(__global cell_t * in, __global cell_t * out,__global bool * change){

    int y = get_global_id (1);
    int x = get_global_id (0);
    
    if(x>0 && y> 0 && x< DIM && y<DIM){ 
        unsigned  n = 0;
        unsigned me  = in[table(y,x)] != 0;

        n += in[table(y,x)];
        n += in[table(y-1,x)];
        n += in[table(y,x-1)];
        n += in[table(y-1,x-1)];
        n += in[table(y+1,x)];
        n += in[table(y,x+1)];
        n += in[table(y+1,x+1)];
        n += in[table(y+1,x-1)];
        n += in[table(y-1,x+1)];    

        n = (n == 3 + me) | (n == 3);
        
        if (n != me)
            *change = 1;
        
        out[table(y,x)]=n;
    }
}


__kernel void life_ocl2(__global cell_t * in, __global cell_t * out,__global bool * change){

    short y = get_group_id (1);
    short x = get_group_id (0);
    
    char yloc = get_local_id (1);
    char xloc = get_local_id (0);

    local char ntab[3][3];

    if(x>0 && y> 0 && x< DIM && y<DIM){ 

        ntab[xloc][yloc]=in[table(y+(2-yloc-1),x+(2-xloc-1))];

        for(int i = 1 ; i < 3; i++){ // reduction des collones
            barrier (CLK_LOCAL_MEM_FENCE);
            if(xloc == 0){
                ntab[xloc][yloc]+=ntab[i][yloc];
            }
        }

         // reduction des lignes
        barrier (CLK_LOCAL_MEM_FENCE);
        if(xloc ==0 && yloc == 0){
            unsigned me = in[table(y,x)];
            unsigned n = 0;  
            n += ntab[0][0];
            n += ntab[0][1];
            n += ntab[0][2];
            n = (n == 3 + me) | (n == 3);
            out[table(y,x)]=n;
            if(me!=n)
                *change |= 1;
        }
    }
    
}



__kernel void life_update_texture (__global unsigned *cur, __write_only image2d_t tex)
{
    int y = get_global_id (1);
    int x = get_global_id (0);
    int2 pos = (int2)(x, y);
    unsigned c = cur [y * DIM + x];

    c = rgba(R*c, G*c, B*c, 0xFF);

    write_imagef (tex, pos, color_scatter (c));
}
