#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <omp.h>

int main()
{
  #pragma omp parallel num_threads(1)
  {
    int me = omp_get_thread_num();

    #pragma omp task firstprivate(me)
    printf("Bonjour de la part de %d exécuté par %d\n",omp_get_thread_num(),me );

    #pragma omp task firstprivate(me)
    printf("Au revoir de la part de %d exécuté par %d\n",omp_get_thread_num(), me );    
  }
  return 0;
}
