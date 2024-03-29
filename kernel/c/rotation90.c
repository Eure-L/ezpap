
#include "easypap.h"

#include <omp.h>
#include <stdbool.h>

#ifdef ENABLE_VECTO
#include <immintrin.h>
#endif

///////////////////////////// Simple sequential version (seq)
// Suggested cmdline:
// ./run --load-image images/shibuya.png --kernel rotation90 --pause
//
unsigned rotation90_compute_seq (unsigned nb_iter)
{
  for (unsigned it = 1; it <= nb_iter; it++) {

    for (int i = 0; i < DIM; i++)
      for (int j = 0; j < DIM; j++)
        next_img (DIM - i - 1, j) = cur_img (j, i);

    swap_images ();
  }
  return 0;
}


static void do_tile_reg (int x, int y, int width, int height)
{
  for (int i = y; i < y + height; i++)
    for (int j = x; j < x + width; j++)
      next_img (DIM-j-1, i) = cur_img (i, j);
}

static inline void do_tile (int x, int y, int width, int height, int who)
{
  monitoring_start_tile (who);

  do_tile_reg (x, y, width, height);

  monitoring_end_tile (x, y, width, height, who);
}

static void do_tile_reg_2 (int x, int y, int width, int height)
{
  #pragma omp parallel for schedule (static)
  for (int i = y; i < y + height; i++)
    for (int j = x; j < x + width; j++)
      next_img (DIM-j-1, i) = cur_img (i, j);
}

static inline void do_tile_2 (int x, int y, int width, int height, int who)
{
  monitoring_start_tile (who);

  do_tile_reg_2 (x, y, width, height);

  monitoring_end_tile (x, y, width, height, who);
}

///////////////////////////// Simple sequential version (seq)
// Suggested cmdline:
// ./run --load-image images/shibuya.png -k rotation90 -v tiled -i 20 -ts 16
//
unsigned rotation90_compute_tiled (unsigned nb_iter)
{
   
  for (unsigned it = 1; it <= nb_iter; it++) {

    for (int y = 0; y < DIM; y += TILE_H)
      for (int x = 0; x < DIM; x += TILE_W)
        do_tile_reg (x, y, TILE_W, TILE_H);
    swap_images ();
  }
  return 0;
}

///////////////////////////// Simple sequential version (seq)
// Suggested cmdline:
// ./run --load-image images/shibuya.png -k rotation90 -v omp_tiled -i 20 -ts 16 -n -t
//
unsigned rotation90_compute_omp_tiled (unsigned nb_iter)
{
  
  for (unsigned it = 1; it <= nb_iter; it++) {
    #pragma omp parallel for collapse(2) schedule(runtime) 
    for (int y = 0; y < DIM; y += TILE_H)
      for (int x = 0; x < DIM; x += TILE_W)
        do_tile (x, y, TILE_W, TILE_H,omp_get_thread_num());
    swap_images ();
  }
  return 0;
}

///////////////////////////// Simple sequential version (seq)
// Suggested cmdline:
// ./run --load-image images/shibuya.png -k rotation90 -v omp_task -i 20 -ts 16 -n -t
//
unsigned rotation90_compute_omp_cache (unsigned nb_iter)
{
  // int largeur = (DIM/2)/omp_get_num_threads();
  int hauteur = DIM/2;
  for (unsigned it = 1; it <= nb_iter; it++) {
    #pragma omp parallel for schedule(static)// collapse(2)
    for (int y = 0; y < DIM/2; y += TILE_H)
      for (int x = 0; x < DIM/2; x += TILE_W){
        
        do_tile (DIM-y-TILE_H, x, TILE_W, TILE_H ,omp_get_thread_num());
        do_tile (x, y,  TILE_W, TILE_H ,omp_get_thread_num());
        do_tile (y, DIM-x-TILE_W, TILE_W, TILE_H ,omp_get_thread_num());
        do_tile (DIM-x-TILE_H, DIM-y-TILE_W,  TILE_W, TILE_H ,omp_get_thread_num());
      }

    swap_images ();
  }
  return 0;
}
