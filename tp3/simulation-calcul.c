#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <omp.h>
#include <sys/time.h>

#define TIME_DIFF(t1, t2) \
  ((t2.tv_sec - t1.tv_sec) * 1000000 + (t2.tv_usec - t1.tv_usec))

int f(int i){
  sleep(1);
  return 2*i;
}

int g(int i){
  sleep(1);
  return 2 * i + 1;
}

int main()
{
  unsigned long temps;
  struct timeval t1, t2;
    gettimeofday(&t1, NULL);

  int x,y;
  #pragma omp parallel
  {
    #pragma omp single nowait
    #pragma omp task shared(x) 
    x = f(2);
    #pragma omp single nowait
    #pragma omp task shared(y) 
    y = g(3);
  }
  printf("résultat %d\n", x+y);

  gettimeofday(&t2, NULL);
  temps = TIME_DIFF(t1, t2);
  fprintf(stderr, "%ld.%03ld\n", temps / 1000, temps % 1000);

  return 0;
}
