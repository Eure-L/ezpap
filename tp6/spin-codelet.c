
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
///////////////////////////////////////////////////////////////////////////



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
  
  // FIXME: we go back to sequential mode here :(
  for (int i = 0; i < VEC_SIZE; i++)
    res[i] = x[i] * M_PI / 4 + 0.273 * x[i] * (1 - fabsf(x[i]));

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
  //__m256 pi2 = _mm256_set1_ps (M_PI_2);

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
  for (int i = 0; i < VEC_SIZE; i++) {
    if (mask[i])
      th[i] = M_PI_2 - th[i];
    if (x[i] < 0)
      th[i] = M_PI - th[i];
    if (y[i] < 0)
      th[i] = -th[i];
  }

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
    for (int j = x; j < x + width; j += VEC_SIZE) {

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
