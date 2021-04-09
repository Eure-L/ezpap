
#include "easypap.h"

#include <omp.h>
#include <mpi.h>


static unsigned compute_one_pixel (int i, int j);
static void zoom (void);

//hello pap

///////////////////////////// Simple sequential version (seq)
// Suggested cmdline:
// ./run --kernel mandel
//
unsigned mandel_compute_seq (unsigned nb_iter)
{
  for (unsigned it = 1; it <= nb_iter; it++) {

    for (int i = 0; i < DIM; i++)
      for (int j = 0; j < DIM; j++)
        cur_img (i, j) = compute_one_pixel (i, j);

    zoom ();
  }

  return 0;
}
///////////////////////////// Simple parallel version (omp)
// Suggested cmdline:
// ./run --kernel mandel -v omp
//
unsigned mandel_compute_omp (unsigned nb_iter)
{
  for (unsigned it = 1; it <= nb_iter; it++) {

    #pragma omp parallel for collapse(2) schedule(runtime)
    for (int i = 0; i < DIM; i++)
      for (int j = 0; j < DIM; j++)
        cur_img (i, j) = compute_one_pixel (i, j);

    zoom ();
  }

  return 0;
}



// Tile inner computation
static void do_tile_reg (int x, int y, int width, int height)
{
  for (int i = y; i < y + height; i++)
    for (int j = x; j < x + width; j++)
      cur_img (i, j) = compute_one_pixel (i, j);
}


///////////////////////////////////////////////////////////////////////////
// Copy this first part and insert into mandel.c, before the do_tile function
#if defined(ENABLE_VECTO) && (AVX2 == 1)

static void do_tile_vec (int x, int y, int width, int height, int who);

#else

#ifdef ENABLE_VECTO
#warning Only 256bit AVX (VEC_SIZE_FLOAT=8) vectorization is currently supported
#endif

#define do_tile_vec(x, y, w, h, who) do_tile ((x), (y), (w), (h), (who))

#endif


static inline void do_tile (int x, int y, int width, int height, int who)
{
  monitoring_start_tile (who);

  do_tile_reg (x, y, width, height);

  monitoring_end_tile (x, y, width, height, who);
}

///////////////////////////// Tiled sequential version (tiled)
// Suggested cmdline:
// ./run -k mandel -v tiled -ts 64
//
unsigned mandel_compute_tiled (unsigned nb_iter)
{
  for (unsigned it = 1; it <= nb_iter; it++) {

    for (int y = 0; y < DIM; y += TILE_H)
      for (int x = 0; x < DIM; x += TILE_W)
        do_tile (x, y, TILE_W, TILE_H, 0);

    zoom ();
  }

  return 0;
}

///////////////////////////// Tiled parallel version (omp)
// Suggested cmdline:
// ./run -k mandel -v tiled -ts 64
//
unsigned mandel_compute_tiled_omp (unsigned nb_iter)
{
  for (unsigned it = 1; it <= nb_iter; it++) {

    #pragma omp parallel for collapse(2) schedule(runtime)
    for (int y = 0; y < DIM; y += TILE_H)
      for (int x = 0; x < DIM; x += TILE_W)
        do_tile (x, y, TILE_W, TILE_H, omp_get_thread_num());
      
    zoom ();
  }

  return 0;
}

///////////////////////////// Vectorized sequential version (vec)
// Suggested cmdline(s):
// ./run -k mandel -v vec
//
unsigned mandel_compute_vec (unsigned nb_iter)
{
  
  for (unsigned it = 1; it <= nb_iter; it++) {

    do_tile_vec (0, 0, DIM, DIM, omp_get_thread_num());

    zoom ();
  }

  return 0;
}

///////////////////////////// Vectorized sequential version (vec)
// Suggested cmdline(s):
// ./run -k mandel -v vec
//
unsigned mandel_compute_ompvec (unsigned nb_iter)
{
  
  for (unsigned it = 1; it <= nb_iter; it++) {
    #pragma omp parallel for collapse(2) schedule(runtime)
    for(int j =0;j<DIM;j+=TILE_H)
      for(int i=0;i<DIM;i+=TILE_W)
        do_tile_vec (i, j, TILE_W, TILE_H, omp_get_thread_num());

    zoom ();
  }

  return 0;
}

///////////////////////////////////////////////////////////////////////////



/////////////// Mandelbrot basic computation

#define MAX_ITERATIONS 4096
#define ZOOM_SPEED -0.01

static float leftX   = -0.2395;
static float rightX  = -0.2275;
static float topY    = .660;
static float bottomY = .648;

static float xstep;
static float ystep;

void mandel_init ()
{
  // check tile size's conformity with respect to CPU vector width
  // easypap_check_vectorization (VEC_TYPE_FLOAT, DIR_HORIZONTAL);

  xstep = (rightX - leftX) / DIM;
  ystep = (topY - bottomY) / DIM;
}

static unsigned iteration_to_color (unsigned iter)
{
  unsigned r = 0, g = 0, b = 0;

  if (iter < MAX_ITERATIONS) {
    if (iter < 64) {
      r = iter * 2; /* 0x0000 to 0x007E */
    } else if (iter < 128) {
      r = (((iter - 64) * 128) / 126) + 128; /* 0x0080 to 0x00C0 */
    } else if (iter < 256) {
      r = (((iter - 128) * 62) / 127) + 193; /* 0x00C1 to 0x00FF */
    } else if (iter < 512) {
      r = 255;
      g = (((iter - 256) * 62) / 255) + 1; /* 0x01FF to 0x3FFF */
    } else if (iter < 1024) {
      r = 255;
      g = (((iter - 512) * 63) / 511) + 64; /* 0x40FF to 0x7FFF */
    } else if (iter < 2048) {
      r = 255;
      g = (((iter - 1024) * 63) / 1023) + 128; /* 0x80FF to 0xBFFF */
    } else {
      r = 255;
      g = (((iter - 2048) * 63) / 2047) + 192; /* 0xC0FF to 0xFFFF */
    }
  }
  return rgba (r, g, b, 255);
}

static void zoom (void)
{
  float xrange = (rightX - leftX);
  float yrange = (topY - bottomY);

  leftX += ZOOM_SPEED * xrange;
  rightX -= ZOOM_SPEED * xrange;
  topY -= ZOOM_SPEED * yrange;
  bottomY += ZOOM_SPEED * yrange;

  xstep = (rightX - leftX) / DIM;
  ystep = (topY - bottomY) / DIM;
}

static unsigned compute_one_pixel (int i, int j)
{
  float cr = leftX + xstep * j;
  float ci = topY - ystep * i;
  float zr = 0.0, zi = 0.0;

  int iter;

  // Pour chaque pixel, on calcule les termes d'une suite, et on
  // s'arrête lorsque |Z| > 2 ou lorsqu'on atteint MAX_ITERATIONS
  for (iter = 0; iter < MAX_ITERATIONS; iter++) {
    float x2 = zr * zr;
    float y2 = zi * zi;

    /* Stop iterations when |Z| > 2 */
    if (x2 + y2 > 4.0)
      break;

    float twoxy = (float)2.0 * zr * zi;
    /* Z = Z^2 + C */
    zr = x2 - y2 + cr;
    zi = twoxy + ci;
  }

  return iteration_to_color (iter);
}

#if defined(ENABLE_VECTO) && (AVX2 == 1)
#include <immintrin.h>

static void compute_multiple_pixels (int i, int j)
{
  __m256 zr, zi, cr, ci, norm; //, iter;
  __m256 deux     = _mm256_set1_ps (2.0);
  __m256 max_norm = _mm256_set1_ps (4.0);

  __m256i iter = _mm256_setzero_si256 ();
  __m256i un   = _mm256_set1_epi32 (1);
  //__m256i vrai = _mm256_set1_epi32 (-1);

  zr = zi = norm = _mm256_set1_ps (0);

  cr = _mm256_add_ps (_mm256_set1_ps (j),
                      _mm256_set_ps (7, 6, 5, 4, 3, 2, 1, 0));

  cr = _mm256_fmadd_ps (cr, _mm256_set1_ps (xstep), _mm256_set1_ps (leftX));

  ci = _mm256_set1_ps (topY - ystep * i);

  for (int i = 0; i < MAX_ITERATIONS; i++) {
    // rc = zr^2
    __m256 rc = _mm256_mul_ps (zr, zr);

    // |Z|^2 = (partie réelle)^2 + (partie imaginaire)^2 = zr^2 + zi^2
    norm = _mm256_fmadd_ps (zi, zi, rc);

    // On compare les normes au carré de chacun des 8 nombres Z avec 4
    // (normalement on doit tester |Z| <= 2 mais c'est trop cher de calculer
    //  une racine carrée)
    // Le résultat est un vecteur d'entiers (mask) qui contient FF quand c'est
    // vrai et 0 sinon
    __m256i mask = (__m256i)_mm256_cmp_ps (norm, max_norm, _CMP_LE_OS);

    // Il faut sortir de la boucle lorsque le masque ne contient que
    // des zéros (i.e. tous les Z ont une norme > 2, donc la suite a
    // divergé pour tout le monde)

    // FIXME 1
    if( _mm256_testz_si256(_mm256_xor_si256(mask,_mm256_setzero_si256()), _mm256_set1_epi32(1))){
      //printf("%d\n",mask);
      break;
    }
    // On met à jour le nombre d'itérations effectuées pour chaque pixel.

    
    // FIXME 2

    iter = _mm256_add_epi32 (iter, _mm256_and_si256(mask,un));

    // On calcule Z = Z^2 + C et c'est reparti !
    __m256 x = _mm256_add_ps (rc, _mm256_fnmadd_ps (zi, zi, cr));
    __m256 y = _mm256_fmadd_ps (deux, _mm256_mul_ps (zr, zi), ci);
    zr       = x;
    zi       = y;
  }

  cur_img (i, j + 0) = iteration_to_color (_mm256_extract_epi32 (iter, 0));
  cur_img (i, j + 1) = iteration_to_color (_mm256_extract_epi32 (iter, 1));
  cur_img (i, j + 2) = iteration_to_color (_mm256_extract_epi32 (iter, 2));
  cur_img (i, j + 3) = iteration_to_color (_mm256_extract_epi32 (iter, 3));
  cur_img (i, j + 4) = iteration_to_color (_mm256_extract_epi32 (iter, 4));
  cur_img (i, j + 5) = iteration_to_color (_mm256_extract_epi32 (iter, 5));
  cur_img (i, j + 6) = iteration_to_color (_mm256_extract_epi32 (iter, 6));
  cur_img (i, j + 7) = iteration_to_color (_mm256_extract_epi32 (iter, 7));
}

static void do_tile_vec (int x, int y, int width, int height,int who)
{
   monitoring_start_tile (who);
  for (int i = y; i < y + height; i++)
    for (int j = x; j < x + width; j += VEC_SIZE_FLOAT)
      compute_multiple_pixels (i, j);

       


    monitoring_end_tile (x, y, width, height, who);
}

#endif
///////////////////////////////////////////////////////////////////////////

///////////////////////////// OpenCL version (ocl)
// Suggested cmdline:
// ./run -k mandel -o
//
unsigned mandel_invoke_ocl (unsigned nb_iter)
{
  size_t global[2] = {GPU_SIZE_X, GPU_SIZE_Y};
  size_t local[2]  = {GPU_TILE_W, GPU_TILE_H};
  cl_int err;
  unsigned max_iter = MAX_ITERATIONS;

  monitoring_start_tile (easypap_gpu_lane (TASK_TYPE_COMPUTE));

  for (unsigned it = 1; it <= nb_iter; it++) {

    // Set kernel arguments
    //
    err = 0;
    err |= clSetKernelArg (compute_kernel, 0, sizeof (cl_mem), &cur_buffer);
    err |= clSetKernelArg (compute_kernel, 1, sizeof (float), &leftX);
    err |= clSetKernelArg (compute_kernel, 2, sizeof (float), &xstep);
    err |= clSetKernelArg (compute_kernel, 3, sizeof (float), &topY);
    err |= clSetKernelArg (compute_kernel, 4, sizeof (float), &ystep);
    err |= clSetKernelArg (compute_kernel, 5, sizeof (unsigned), &max_iter);

    check (err, "Failed to set kernel arguments");

    err = clEnqueueNDRangeKernel (queue, compute_kernel, 2, NULL, global, local,
                                  0, NULL, NULL);
    check (err, "Failed to execute kernel");

    zoom ();
  }

  clFinish (queue);

  monitoring_end_tile (0, 0, DIM, DIM, easypap_gpu_lane (TASK_TYPE_COMPUTE));

  return 0;
}

///////////////////////////// OpenCL bybrid version (ocl_hybrid)

// Threashold = 10%
#define THRESHOLD 10

static unsigned cpu_y_part;
static unsigned gpu_y_part;

void mandel_init_ocl_hybrid (void)
{
  if (GPU_TILE_H != TILE_H)
    exit_with_error ("CPU and GPU Tiles should have the same height (%d != %d)",
                     GPU_TILE_H, TILE_H);

  cpu_y_part = (NB_TILES_Y / 2) * GPU_TILE_H; // Start with fifty-fifty
  gpu_y_part = DIM - cpu_y_part;
}

static long gpu_duration = 0, cpu_duration = 0;

static int much_greater_than (long t1, long t2)
{
  return (t1 > t2) && ((t1 - t2) * 100 / t1 > THRESHOLD);
}

unsigned mandel_invoke_ocl_hybrid (unsigned nb_iter)
{
  size_t global[2] = {DIM, gpu_y_part};
  size_t local[2]  = {GPU_TILE_W, GPU_TILE_H};
  cl_int err;
  unsigned max_iter = MAX_ITERATIONS;
  cl_event kernel_event;
  long t1, t2;
  int gpu_accumulated_lines = 0;

  for (unsigned it = 1; it <= nb_iter; it++) {
    
    // Load balancing
    if (gpu_duration) {
      if (much_greater_than (gpu_duration, cpu_duration) &&
          gpu_y_part > GPU_TILE_H) {
        gpu_y_part -= GPU_TILE_H;
        cpu_y_part += GPU_TILE_H;
        global[1] = gpu_y_part;
      } else if (much_greater_than (cpu_duration, gpu_duration) &&
                 cpu_y_part > GPU_TILE_H) {
        cpu_y_part -= GPU_TILE_H;
        gpu_y_part += GPU_TILE_H;
        global[1] = gpu_y_part;
      }
    }

    // Set kernel arguments
    //
    err = 0;
    err |= clSetKernelArg (compute_kernel, 0, sizeof (cl_mem), &cur_buffer);
    err |= clSetKernelArg (compute_kernel, 1, sizeof (float), &leftX);
    err |= clSetKernelArg (compute_kernel, 2, sizeof (float), &xstep);
    err |= clSetKernelArg (compute_kernel, 3, sizeof (float), &topY);
    err |= clSetKernelArg (compute_kernel, 4, sizeof (float), &ystep);
    err |= clSetKernelArg (compute_kernel, 5, sizeof (unsigned), &max_iter);
    err |= clSetKernelArg (compute_kernel, 6, sizeof (unsigned), &cpu_y_part);
    check (err, "Failed to set kernel arguments");

    // Launck GPU kernel
    err = clEnqueueNDRangeKernel (queue, compute_kernel, 2, NULL, global, local,
                                  0, NULL, &kernel_event);
    check (err, "Failed to execute kernel");
    clFlush (queue);

    // Compute CPU part
    t1 = what_time_is_it ();

#pragma omp parallel for collapse(2) schedule(runtime)
    for (int y = 0; y < cpu_y_part; y += TILE_H)
      for (int x = 0; x < DIM; x += TILE_W)
        do_tile (x, y, TILE_W, TILE_H, omp_get_thread_num ());

    t2           = what_time_is_it ();
    cpu_duration = t2 - t1;

    clFinish (queue);

    gpu_duration = ocl_monitor (kernel_event, 0, cpu_y_part, global[0],
                                global[1], TASK_TYPE_COMPUTE);
    clReleaseEvent (kernel_event);

    gpu_accumulated_lines += gpu_y_part;

    zoom ();
  }

  if (do_display) {
    // Send CPU contribution to GPU memory
    err = clEnqueueWriteBuffer (queue, cur_buffer, CL_TRUE, 0,
                                DIM * cpu_y_part * sizeof (unsigned), image, 0,
                                NULL, NULL);
    check (err, "Failed to write to buffer");
  } else
    PRINT_DEBUG ('u', "In average, GPU took %.1f%% of the lines\n",
                 (float)gpu_accumulated_lines * 100 / (DIM * nb_iter));

  return 0;
}

static int rank, size;

void mandel_init_mpi ()
{
  easypap_check_mpi ();// check if MPI was correctly configured

  /* récupérer son propre numéro */
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  /* récupérer le nombre de processus lancés */
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  mandel_init ();
}

void mandel_init_mpi_omp ()
{
  easypap_check_mpi ();// check if MPI was correctly configured

  /* récupérer son propre numéro */
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  /* récupérer le nombre de processus lancés */
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  mandel_init ();

}

static int rankTop (int rank)
{
  return ((rank-1) * (DIM/(size-1)) ); // indice 1ere ligne
}

static int rankSize (int rank)
{
  if(rank < size)
    return ((DIM/(size-1)));// nombre de lignes
  return 0;
}

void mandel_refresh_img_mpi ()
{
  // le maitre réceptionne les données
  // les autres processus envoient les données
  if(rank == 0){

    for(int i=1;i<size;i++){
      MPI_Recv(&cur_img(rankTop(i),0),rankSize(rank)  * DIM ,MPI_INT,i,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
    }
  }

  else{
    MPI_Send(&cur_img(rankTop(rank),0),rankSize (rank)*DIM,MPI_INT,0,0,MPI_COMM_WORLD);
  }
}

void mandel_refresh_img_mpi_omp ()
{
  // le maitre réceptionne les données
  // les autres processus envoient les données
  #if 0
  if(rank == 0){

    for(int i=1;i<size;i++){
      MPI_Recv(&cur_img(rankTop(i),0),rankSize(rank)  * DIM ,MPI_INT,i,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
    }
  }

  else{
    MPI_Send(&cur_img(rankTop(rank),0),rankSize (rank)*DIM,MPI_INT,0,0,MPI_COMM_WORLD);
  }
  #else 

  MPI_Gather((rank== 0) ? MPI_IN_PLACE : &cur_img(rankTop (rank), 0), rankSize(rank) * DIM, MPI_INT,
             &cur_img(0, 0),  rankSize(rank) * DIM, MPI_INT,
             0,  MPI_COMM_WORLD);

  #endif
}

////////// MPI basic variant
// Suggested cmdline:
// ./run -k mandel -v mpi -mpi "-np 4"  -d M

unsigned mandel_compute_mpi (unsigned nb_iter)
{

  for(unsigned it = 1; it <= nb_iter; it++) {
    if(rank!=0)
      do_tile_vec (0, rankTop (rank), DIM, rankSize (rank), rank);
    mandel_refresh_img_mpi();
    zoom();
  }
  return 0;
}

unsigned mandel_compute_mpi_omp (unsigned nb_iter)
{

  for(unsigned it = 1; it <= nb_iter; it++) {
    if(rank!=0){
      #pragma omp parallel for num_threads(8) schedule(dynamic)
      for(int l = rankTop(rank); l < rankTop(rank) + rankSize(rank); l++)
        do_tile_vec (0, l, DIM,1, omp_get_thread_num());
    }
    mandel_refresh_img_mpi();
    zoom();
  }
  return 0;
}
