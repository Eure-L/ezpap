
void tsp_ompcol4()
{
  int i,j,k;
#pragma omp parallel for collapse(4) schedule(runtime)
 for (i=1; i < nbVilles; i++)
   for(j=1; j < nbVilles; j++)
     for(k=1; k < nbVilles; k++)
        for(int l=1; l < nbVilles; l++)
	  if(i != j && i != k && j != k && i != l && j != l && k != l)
	    {
	      chemin_t chemin;
	      chemin[0] = 0;
	      chemin[1] = i;
	      chemin[2] = j;
	      chemin[3] = k;
	      chemin[4] = l;
	      int dist = distance[0][i] + distance[i][j] + distance[j][k] + distance[k][l];
	      tsp_seq (5, dist, chemin,  1 | (1<<i) | (1<<j) | (1<<k) | (1<<l)) ;
         }
}

void tsp_ompcol3()
{
  int i,j,k;
#pragma omp parallel for collapse(3) schedule(runtime)
 for (i=1; i < nbVilles; i++)
   for(j=1; j < nbVilles; j++)
     for(k=1; k < nbVilles; k++)
       if(i != j && i != k && j != k)
         {
          chemin_t chemin;
          chemin[0] = 0;
          chemin[1] = i;
          chemin[2] = j;
          chemin[3] = k;
          int dist = distance[0][i] + distance[i][j] + distance[j][k];
          tsp_seq (4, dist, chemin,  1 | (1<<i) | (1<<j) | (1<<k)) ;
         }
}

void tsp_ompcol2()
{
  int i,j;
#pragma omp parallel for collapse(2) schedule(runtime)
 for (i=1; i < nbVilles; i++)
   for(j=1; j < nbVilles; j++)
     if(i != j)
         {
          chemin_t chemin;
          chemin[0] = 0;
          chemin[1] = i;
          chemin[2] = j;
          int dist = distance[0][i] + distance[i][j];
          tsp_seq (3, dist, chemin,  1 | (1<<i) | (1<<j) );
         }
}
