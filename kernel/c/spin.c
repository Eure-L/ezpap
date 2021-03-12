
#include "easypap.h"

#include <math.h>
#include <omp.h>

static void rotate (void);
static unsigned compute_color (int i, int j);
//#define VEC_SIZE 32

// If defined, the initialization hook function is called quite early in the
// initialization process, after the size (DIM variable) of images is known.
// This function can typically spawn a team of threads, or allocated additionnal
// OpenCL buffers.
// A function named <kernel>_init_<variant> is search first. If not found, a
// function <kernel>_init is searched in turn.
void spin_init (void)
{
  // check tile size's conformity with respect to CPU vector width
  // easypap_check_vectorization (VEC_TYPE_FLOAT, DIR_HORIZONTAL);

  PRINT_DEBUG ('u', "Image size is %dx%d\n", DIM, DIM);
  PRINT_DEBUG ('u', "Block size is %dx%d\n", TILE_W, TILE_H);
  PRINT_DEBUG ('u', "Press <SPACE> to pause/unpause, <ESC> to quit.\n");
}

// The image is a two-dimension array of size of DIM x DIM. Each pixel is of
// type 'unsigned' and store the color information following a RGBA layout (4
// bytes). Pixel at line 'l' and column 'c' in the current image can be accessed
// using cur_img (l, c).

// The kernel returns 0, or the iteration step at which computation has
// completed (e.g. stabilized).

///////////////////////////// Simple sequential version (seq)
// Suggested cmdline(s):
// ./run --size 1024 --kernel spin --variant seq
// or
// ./run -s 1024 -k spin -v seq
//
unsigned spin_compute_seq (unsigned nb_iter)
{
  for (unsigned it = 1; it <= nb_iter; it++) {

    for (int i = 0; i < DIM; i++)
      for (int j = 0; j < DIM; j++)
        cur_img (i, j) = compute_color (i, j);

    rotate (); // Slightly increase the base angle
  }

  return 0;
}


// Tile inner computation
static void do_tile_reg (int x, int y, int width, int height)
{
  for (int i = y; i < y + height; i++)
    for (int j = x; j < x + width; j++)
      cur_img (i, j) = compute_color (i, j);
}

///////////////////////////////////////////////////////////////////////////
// Copy this first part and insert into spin.c, before the do_tile function
#if defined(ENABLE_VECTO) && (AVX2 == 1)

static void do_tile_vec (int x, int y, int width, int height);

#else

#ifdef ENABLE_VECTO
#warning Only 256bit AVX (VEC_SIZE_FLOAT=8) vectorization is currently supported
#endif

#define do_tile_vec(x, y, w, h) do_tile_reg ((x), (y), (w), (h))

#endif


///////////////////////////////////////////////////////////////////////////

static void do_tile (int x, int y, int width, int height, int who)
{
  // Calling monitoring_{start|end}_tile before/after actual computation allows
  // to monitor the execution in real time (--monitoring) and/or to generate an
  // execution trace (--trace).
  // monitoring_start_tile only needs the cpu number
  monitoring_start_tile (who);

  do_tile_reg (x, y, width, height);

  // In addition to the cpu number, monitoring_end_tile also needs the tile
  // coordinates
  monitoring_end_tile (x, y, width, height, who);
}


///////////////////////////// Tiled sequential version (tiled)
// Suggested cmdline(s):
// ./run -k spin -v tiled -ts 64 -m
//
unsigned spin_compute_tiled (unsigned nb_iter)
{
  for (unsigned it = 1; it <= nb_iter; it++) {

    for (int y = 0; y < DIM; y += TILE_H)
      for (int x = 0; x < DIM; x += TILE_W)
        if((x / TILE_W + y / TILE_H) % 2)
        do_tile (x, y, TILE_W, TILE_H, 0 /* CPU id */);

    rotate ();
  }

  return 0;
}

/////////////////////////////  omp version (tiled)
// Suggested cmdline(s):
// ./run -k spin -v tiled -ts 64 -m
//
unsigned spin_compute_omp__(unsigned nb_iter)
{
  for (unsigned it = 1; it <= nb_iter; it++) {
    #pragma omp parallel for schedule(runtime)
    for (int y = 0; y < DIM; y += TILE_H)
      for (int x = 0; x < DIM; x += TILE_W)
        do_tile (x, y, TILE_W, TILE_H, omp_get_thread_num());
    rotate ();
  }

  return 0;
}

/////////////////////////////  omp version (tiled)
// Suggested cmdline(s):
// ./run -k spin -v tiled -ts 64 -m
//
unsigned spin_compute_omp(unsigned nb_iter)
{
  for (unsigned it = 1; it <= nb_iter; it++) {
    #pragma omp parallel for collapse(2) schedule(runtime) 
    for (int y = 0; y < DIM; y += TILE_H)
      for (int x = 0; x < DIM; x += TILE_W)
        do_tile (x, y, TILE_W, TILE_H, omp_get_thread_num());
    rotate ();
  }

  return 0;
}

///////////////////////////// Vectorized sequential version (vec)
// Suggested cmdline(s):
// ./run -k spin -v vec
//
unsigned spin_compute_vec (unsigned nb_iter)
{

  for (unsigned it = 1; it <= nb_iter; it++) {

    do_tile_vec (0, 0, DIM, DIM);

    rotate ();
  }

  return 0;
}

//////////////////////////////////////////////////////////////////////////

static float base_angle = 0.0;
static int color_a_r = 255, color_a_g = 255, color_a_b = 0, color_a_a = 255;
static int color_b_r = 0, color_b_g = 0, color_b_b = 255, color_b_a = 255;

static float atanf_approx (float x)
{
  float a = fabsf (x);

  return x * M_PI / 4 + 0.273 * x * (1 - a);
}

static float atan2f_approx (float y, float x)
{
  float ay   = fabsf (y);
  float ax   = fabsf (x);
  int invert = ay > ax;
  float z    = invert ? ax / ay : ay / ax; // [0,1]
  float th   = atanf_approx (z);           // [0,π/4]
  if (invert)
    th = M_PI_2 - th; // [0,π/2]
  if (x < 0)
    th = M_PI - th; // [0,π]
  if (y < 0)
    th = -th;

  return th;
}

// Computation of one pixel
static unsigned compute_color (int i, int j)
{
  float angle =
      atan2f_approx ((int)DIM / 2 - i, j - (int)DIM / 2) + M_PI + base_angle;

  float ratio = fabsf ((fmodf (angle, M_PI / 4.0) - (float)(M_PI / 8.0)) /
                       (float)(M_PI / 8.0));

  int r = color_a_r * ratio + color_b_r * (1.0 - ratio);
  int g = color_a_g * ratio + color_b_g * (1.0 - ratio);
  int b = color_a_b * ratio + color_b_b * (1.0 - ratio);
  int a = color_a_a * ratio + color_b_a * (1.0 - ratio);

  return rgba (r, g, b, a);
}

static void rotate (void)
{
  base_angle = fmodf (base_angle + (1.0 / 180.0) * M_PI, M_PI);
}

///////////////////////////////////////////////////////////////////////////
// Copy this second part and insert at the end of spin.c

// Intrinsics functions
#if defined(ENABLE_VECTO) && (AVX2 == 1)
#include <immintrin.h>

static inline __m256 _mm256_abs_ps (__m256 a)
{
  __m256i minus1 = _mm256_set1_epi32 (-1);
  __m256 mask    = _mm256_castsi256_ps (_mm256_srli_epi32 (minus1, 1));

  return _mm256_and_ps (a, mask);
}

static __m256 _mm256_atan_ps (__m256 x)
{
  __m256 res;
  __m256 pidiv4 = _mm256_set1_ps (M_PI / 4);
  __m256 k = _mm256_set1_ps(0.273);
  __m256 one = _mm256_set1_ps(1);
  __m256 t = _mm256_sub_ps(one,_mm256_abs_ps(x));
  // FIXME: we go back to sequential mode here :(
  //for (int i = 0; i < VEC_SIZE_FLOAT; i++)
    //res[i] = x[i] * M_PI / 4 + 0.273 * x[i] * (1 - fabsf(x[i]));
  res =  _mm256_add_ps(_mm256_mul_ps(x, pidiv4) , _mm256_mul_ps( _mm256_mul_ps(k, x) , t));
  return res;
  
  //  return x * M_PI / 4 + 0.273 * x * (1 - abs(x));
}

// if mask[i] then result[i] = -a[i] else result = a[i]
static inline __m256 _mm256_negmask_ps (__m256 a, __m256 mask)
{
  __m256 sign = _mm256_castsi256_ps (_mm256_set1_epi32 (0x80000000));

  sign = _mm256_and_ps (sign, mask);

  return _mm256_xor_ps (a, sign);
}

static __m256 _mm256_atan2_ps (__m256 y, __m256 x)
{
  // __m256 pi2 = _mm256_set1_ps (M_PI_2);
  // __m256 pi = _mm256_set1_ps (M_PI);
  // __m256 minusone = _mm256_set1_ps (-1);

  // float ay = fabsf (y), ax = fabsf (x);
  __m256 ax = _mm256_abs_ps (x);
  __m256 ay = _mm256_abs_ps (y);

  // int invert = ay > ax;
  __m256 mask = _mm256_cmp_ps (ay, ax, _CMP_GT_OS);

  // float z    = invert ? ax / ay : ay / ax;
  __m256 top = _mm256_min_ps (ax, ay);
  __m256 bot = _mm256_max_ps (ax, ay);
  __m256 z   = _mm256_div_ps (top, bot);

  // float th = atan_approx(z);
  __m256 th = _mm256_atan_ps (z);

  // FIXME: we go back to sequential mode here :(
  
#if 0
  for (int i = 0; i < VEC_SIZE_FLOAT; i++) {
    if (mask[i])
     th[i] = M_PI_2 - th[i];

    if (x[i] < 0)
      th[i] = M_PI - th[i];

    if (y[i] < 0)
      th[i] = -th[i];

  }
  #else

    th = _mm256_negmask_ps(th,mask);
    th = _mm256_add_ps(th,_mm256_and_ps(mask,_mm256_set1_ps(M_PI_2)));

    mask = _mm256_cmp_ps (x, _mm256_setzero_ps(), _CMP_LT_OS);
    th = _mm256_negmask_ps(th,mask);
    th = _mm256_add_ps(th,_mm256_and_ps(mask,_mm256_set1_ps(M_PI)));

    mask = _mm256_cmp_ps (y, _mm256_setzero_ps(), _CMP_LT_OS);
    th = _mm256_negmask_ps(th,mask);

  #endif

  return th;
}

// We assume a > 0 and b > 0
static inline __m256 _mm256_mod_ps (__m256 a, __m256 b)
{
  __m256 r = _mm256_floor_ps (_mm256_div_ps (a, b));

  // return a - r * b;
  return _mm256_fnmadd_ps (r, b, a);
}

static inline __m256 _mm256_mod2_ps (__m256 a, __m256 b, __m256 invb)
{
  __m256 r = _mm256_floor_ps (_mm256_mul_ps (a, invb));

  // return a - r * b;
  return _mm256_fnmadd_ps (r, b, a);
}

static void do_tile_vec (int x, int y, int width, int height)
{
  __m256 pi4    = _mm256_set1_ps (M_PI / 4);
  __m256 invpi4 = _mm256_set1_ps (4.0 / M_PI);
  __m256 invpi8 = _mm256_set1_ps (8.0 / M_PI);
  __m256 one    = _mm256_set1_ps (1.0);
  __m256 dim2   = _mm256_set1_ps (DIM / 2);
  __m256 ang    = _mm256_set1_ps (base_angle + M_PI);

  for (int i = y; i < y + height; i++)
    for (int j = x; j < x + width; j += VEC_SIZE_FLOAT) {

      __m256 vi = _mm256_set1_ps (i);
      __m256 vj = _mm256_add_ps (_mm256_set1_ps (j),
                                 _mm256_set_ps (7, 6, 5, 4, 3, 2, 1, 0));

      // float angle = atan2f_approx ((int)DIM / 2 - i, j - (int)DIM / 2) + M_PI
      // + base_angle;
      __m256 angle =
          _mm256_atan2_ps (_mm256_sub_ps (dim2, vi), _mm256_sub_ps (vj, dim2));

      angle = _mm256_add_ps (angle, ang);

      // float ratio = fabsf ((fmodf (angle, M_PI / 4.0) - (float)(M_PI / 8.0))
      // / (float)(M_PI / 8.0));
      __m256 ratio = _mm256_mod2_ps (angle, pi4, invpi4);

      ratio = _mm256_fmsub_ps (ratio, invpi8, one);
      ratio = _mm256_abs_ps (ratio);

      __m256 ratiocompl = _mm256_sub_ps (one, ratio);

      __m256 red = _mm256_mul_ps (_mm256_set1_ps (color_a_r), ratio);
      red = _mm256_fmadd_ps (_mm256_set1_ps (color_b_r), ratiocompl, red);

      __m256 green = _mm256_mul_ps (_mm256_set1_ps (color_a_g), ratio);
      green = _mm256_fmadd_ps (_mm256_set1_ps (color_b_g), ratiocompl, green);

      __m256 blue = _mm256_mul_ps (_mm256_set1_ps (color_a_b), ratio);
      blue = _mm256_fmadd_ps (_mm256_set1_ps (color_b_b), ratiocompl, blue);

      __m256 alpha = _mm256_mul_ps (_mm256_set1_ps (color_a_a), ratio);
      alpha = _mm256_fmadd_ps (_mm256_set1_ps (color_b_a), ratiocompl, alpha);

      __m256i color = _mm256_cvtps_epi32 (alpha);

      color = _mm256_or_si256 (
          color, _mm256_slli_epi32 (_mm256_cvtps_epi32 (blue), 8));
      color = _mm256_or_si256 (
          color, _mm256_slli_epi32 (_mm256_cvtps_epi32 (green), 16));
      color = _mm256_or_si256 (
          color, _mm256_slli_epi32 (_mm256_cvtps_epi32 (red), 24));

      _mm256_store_si256 ((__m256i *)&cur_img (i, j), color);
    }
}

#endif
///////////////////////////////////////////////////////////////////////////
