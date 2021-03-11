
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
// ./run -k mandel -v vec
//
unsigned mandel_compute_vec (unsigned nb_iter)
{

  for (unsigned it = 1; it <= nb_iter; it++) {

    do_tile_vec (0, 0, DIM, DIM);

    zoom ();
  }

  return 0;
}
///////////////////////////////////////////////////////////////////////////



///////////////////////////////////////////////////////////////////////////
// Copy this second part and insert at the end of spin.c

#if defined(ENABLE_VECTO) && (AVX2 == 1)
#include <immintrin.h>

static void compute_multiple_pixels (int i, int j)
{
  __m256 zr, zi, cr, ci, norm; //, iter;
  __m256 deux     = _mm256_set1_ps (2.0);
  __m256 max_norm = _mm256_set1_ps (4.0);

  __m256i iter = _mm256_setzero_si256 ();
  __m256i un   = _mm256_set1_epi32 (1);
  __m256i vrai = _mm256_set1_epi32 (-1);

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

    // On met à jour le nombre d'itérations effectuées pour chaque pixel.

    // FIXME 2
    iter = _mm256_add_epi32 (iter, un);

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

static void do_tile_vec (int x, int y, int width, int height)
{
  for (int i = y; i < y + height; i++)
    for (int j = x; j < x + width; j += VEC_SIZE)
      compute_multiple_pixels (i, j);
}

#endif
///////////////////////////////////////////////////////////////////////////
