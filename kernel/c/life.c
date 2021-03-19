#include "easypap.h"
#include "rle_lexer.h"
#include "arch_flags.h"

#include <avxintrin.h>
#include <omp.h>
#include <stdbool.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

#define NB_TILES_TOT (NB_TILES_X*NB_TILES_Y)



static unsigned color = 0xFFFF00FF; // Living cells have the yellow color

typedef char cell_t;

static cell_t *restrict _table = NULL, *restrict _alternate_table = NULL;

bool init = true;
bool switcher = 1;
unsigned curr_tasks = 1;
unsigned next_tasks = 0;
omp_lock_t * writelock;
omp_lock_t * changeLock;

#define curTable switcher
#define nextTable !switcher

char * bitMapTls; // Two Bit maps represented in a vector
                  // Each bits representing a tile to compute


static inline cell_t *table_cell (cell_t *restrict i, int y, int x)
{
  return i + y * DIM + x;

}

static inline char *table_map (char *restrict i, int y, int x)
{
  return i + y * NB_TILES_X + x;
}


// This kernel does not directly work on cur_img/next_img.
// Instead, we use 2D arrays of boolean values, not colors
#define cur_table(y, x) (*table_cell (_table, (y), (x)))
#define next_table(y, x) (*table_cell (_alternate_table, (y), (x)))

#define cur_map(y, x) (*table_map ((bitMapTls+(curTable*NB_TILES_TOT)), (y), (x)))
#define next_map(y, x) (*table_map ((bitMapTls+(nextTable*NB_TILES_TOT)), (y), (x)))

#define right 1
#define left 2
#define top 4
#define topright 5
#define topleft 6
#define bot 8
#define botright 9
#define botleft 10
#define botrightleft 11
#define righttopbot 13
#define lefttopbot 14
#define rightlefttopbot 15

////
bool isOnLeft(int x,int y){
  return x > 0 && x%TILE_W == 0;
}
bool isOnRight(int x,int y){
  return (x<DIM-1 && (x%TILE_W == TILE_W-1));
}
bool isOnTop(int x,int y){
  return (y>0 && y%TILE_H==0);
}
bool isOnBottom(int x,int y){
  return (y<DIM-1 && (y%TILE_H == TILE_H-1));
}
//
bool isTileOnLeft(int i,int j){
  return i == 0;
}
bool isTileOnRight(int i,int j){
  return i == NB_TILES_X-1;
}
bool isTileOnTop(int i,int j){
  return j == 0;
}
bool isTileOnBottom(int i,int j){
  return j == NB_TILES_Y-1;
}
////


char * initBtmptls(omp_lock_t * lock,int curr_tasks){
  
  char * map = (char *) malloc ((2 * NB_TILES_X * NB_TILES_Y) * sizeof(char));
  omp_init_lock(lock);
  if(map == NULL){
    printf("map pointer NULL\n");
    exit(EXIT_FAILURE);
  }       

  //int idMap;
  for (int i=0;i<NB_TILES_TOT;i++){
    *(map+i)=0;
  }
  for (int i=0;i<NB_TILES_TOT;i++){
    *(map+NB_TILES_TOT+i)=1;
  }
    //printf("init a : %d \n",*(map+(NB_TILES_TOT)+i));
  
  return map;
}

char * initInnertls(void){
  
  char * map = (char *) malloc (( NB_TILES_X * NB_TILES_Y) * sizeof(char));
  if(map == NULL){
    printf("map pointer NULL\n");
    exit(EXIT_FAILURE);
  }    

  for (int i=0;i<NB_TILES_TOT;i++){
    *(map+i)=0;
  }
  
  return map;
}

void addTaskBtmp( int i, int j,char * map){
  //if(i>=0 && j>=0 && i<NB_TILES_X && j< NB_TILES_Y){
     *(map+j*(NB_TILES_X)+i)=1;
  //}
  //return 0;
}

void printBitmaps(char * btmp,bool current){

  printf("Current bitmap : \n");
  for(int i = 0; i<NB_TILES_X; i++){
    for(int j = 0; j<NB_TILES_Y; j++){
      printf(" %d ", *(btmp+current*NB_TILES_TOT+j*(NB_TILES_X)+i));
    }
    printf("\n");
  }
  printf("\n Next bitmap : \n");
  for(int i = 0; i<NB_TILES_X; i++){
    for(int j = 0; j<NB_TILES_Y; j++){
      printf(" %d ",*(btmp+(!current)*NB_TILES_TOT+j*(NB_TILES_X)+i));
    }
    printf("\n");
  }
}
void printInnermap(char * btmp){

  printf("Inner bitmap : \n");
  for(int i = 0; i<NB_TILES_X; i++){
    for(int j = 0; j<NB_TILES_Y; j++){
      printf(" %d ", *(btmp+j*(NB_TILES_X)+i));
    }
    printf("\n");
  }
}

void deleteBtmp(char * btmp){
  for(int i = 0; i<NB_TILES_X; i++){
    for(int j = 0;j<NB_TILES_Y; j++){
      *(btmp+j*(NB_TILES_X)+i)=0;
    }
  }
}

unsigned tilePosition(int i, int j){
  unsigned mask = 0;
  mask |= top * isTileOnTop(i,j) | bot * isTileOnBottom(i,j) | \
          left * isTileOnLeft(i,j) | right * isTileOnRight(i,j);
  return mask;
}


void prntAVXi( __m256i vec,char * name){
  int byteSize = sizeof(char);
  const size_t n = sizeof(__m256i) / byteSize;
  char buffer[n];
  _mm256_storeu_si256((void*)buffer, vec);
  int i = 0;
  printf("-- %s -- \n",name);
  for (; i < 32 ; i++){
      //if(buffer[i]!=0)
         
      printf("%u.",buffer[i]);
      
  }
  printf(" ~ i : %d\n",i);
}


void life_init (void)
{
  // life_init may be (indirectly) called several times so we check if data were
  // already allocated
  if (_table == NULL) {
    const unsigned size = DIM * DIM * sizeof (cell_t);

    PRINT_DEBUG ('u', "Memory footprint = 2 x %d bytes\n", size);

    _table = mmap (NULL, size, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);

    _alternate_table = mmap (NULL, size, PROT_READ | PROT_WRITE,
                             MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  }
}

void life_finalize (void)
{
  const unsigned size = DIM * DIM * sizeof (cell_t);

  munmap (_table, size);
  munmap (_alternate_table, size);
}

// This function is called whenever the graphical window needs to be refreshed
void life_refresh_img (void)
{
  for (int i = 0; i < DIM; i++)
    for (int j = 0; j < DIM; j++)
      cur_img (i, j) = cur_table (i, j) * color;
}

static inline void swap_tables (void)
{
  cell_t *tmp = _table;

  _table           = _alternate_table;
  _alternate_table = tmp;
}

///////////////////////////// Sequential version (seq)

static int compute_new_state (int y, int x)
{
  unsigned n      = 0;
  unsigned me     = cur_table (y, x) != 0;
  unsigned change = 0;

  if (x > 0 && x < DIM - 1 && y > 0 && y < DIM - 1) {

    for (int i = y - 1; i <= y + 1; i++)
      for (int j = x - 1; j <= x + 1; j++)
        n += cur_table (i, j);

    n = (n == 3 + me) | (n == 3);
    if (n != me)
      change |= 1;

    next_table (y, x) = n;
  }

  return change;
}

unsigned life_compute_seq (unsigned nb_iter)
{
  int change = 0;
  unsigned it = 1;
  for (; it <= nb_iter; it++) {
    

    monitoring_start_tile (0);

    for (int i = 0; i < DIM; i++)
      for (int j = 0; j < DIM; j++)
        change |= compute_new_state (i, j);

    monitoring_end_tile (0, 0, DIM, DIM, 0);

    swap_tables ();

    if (!change)
      return it;
  }
  return 0 ;
}


///////////////////////////// Tiled sequential version (tiled)

// Tile inner computation
static int do_tile_reg (int x, int y, int width, int height)
{
  int change = 0;

  for (int i = y; i < y + height; i++)
    for (int j = x; j < x + width; j++)
      change |= compute_new_state (i, j);

  return change;
}

static int do_tile (int x, int y, int width, int height, int who)
{
  int r;

  monitoring_start_tile (who);

  r = do_tile_reg (x, y, width, height);

  monitoring_end_tile (x, y, width, height, who);

  return r;
}

unsigned life_compute_tiled (unsigned nb_iter)
{
  unsigned res = 0;

  for (unsigned it = 1; it <= nb_iter; it++) {
    unsigned change = 0;

    for (int y = 0; y < DIM; y += TILE_H)
      for (int x = 0; x < DIM; x += TILE_W)
        change |= do_tile (x, y, TILE_W, TILE_H, 0);

    swap_tables ();

    if (!change) { // we stop when all cells are stable
      res = it;
      break;
    }
  }

  return res;
}

//////////////////////////////// Tiled omp version
// ./run -k random -s 1024 -ts 128 -i 100 -n -v omp
unsigned life_compute_omp (unsigned nb_iter)
{

  unsigned res = 0;

  for (unsigned it = 1; it <= nb_iter; it++) {
    unsigned change = 0;
    #pragma omp parallel for collapse(2) schedule (static)
    for (int y = 0; y < DIM; y += TILE_H)
      for (int x = 0; x < DIM; x += TILE_W)
        change |= do_tile (x, y, TILE_W, TILE_H, omp_get_thread_num());

    swap_tables ();
    
    if (!change) { // we stop when all cells are stable
      res = it;
      break;
    }
  }
  return res;
}


static int compute_new_state_border (int y, int x)
{
  unsigned n      = 0;
  unsigned me     = cur_table (y, x) != 0;
  unsigned change = 0;

  if (x > 0 && x < DIM - 1 && y > 0 && y < DIM - 1) {

    for (int i = y - 1; i <= y + 1; i++)
      for (int j = x - 1; j <= x + 1; j++)
        n += cur_table (i, j);

    n = (n == 3 + me) | (n == 3);
    if (n != me)
      change |= 1;

    next_table (y, x) = n;
    
    //preloading potential load for all cells on the border of the tiles
    if(change == 1){
      bool isleft = isOnLeft(x,y);
      bool isright = isOnRight(x,y);
      bool istop = isOnTop(x,y);
      bool isbottom = isOnBottom(x,y);
      if(isleft){
        addTaskBtmp((x-TILE_W)/TILE_W,(y/TILE_H),bitMapTls+nextTable*NB_TILES_TOT);    
        if(istop)
          addTaskBtmp((x-TILE_W)/TILE_W,(y-TILE_H)/TILE_H,bitMapTls+nextTable*NB_TILES_TOT);
        else if(isbottom)
          addTaskBtmp((x-TILE_W)/TILE_W,(y+1)/TILE_H,bitMapTls+nextTable*NB_TILES_TOT);
      }
      else if (isright){
        addTaskBtmp((x+1)/TILE_W,(y/TILE_H),bitMapTls+nextTable*NB_TILES_TOT);  
        if(istop)
          addTaskBtmp((x+1)/TILE_W,(y-TILE_H)/TILE_H,bitMapTls+nextTable*NB_TILES_TOT);
        else if(isbottom)
          addTaskBtmp((x+1)/TILE_W,(y+1)/TILE_H,bitMapTls+nextTable*NB_TILES_TOT);   
      }
      if(istop){
        addTaskBtmp((x/TILE_W),(y-TILE_H)/TILE_H,bitMapTls+nextTable*NB_TILES_TOT);   
      }
      else if (isbottom){
        addTaskBtmp((x/TILE_W),(y+1)/TILE_H,bitMapTls+nextTable*NB_TILES_TOT);
      }
    }
  }

  return change;
}

static int do_tile_reg_border (int x, int y, int width, int height)
{
  int change = 0;

  for (int i = y; i < y + height; i++){
    for (int j = x; j < x + width; j++){
      change |= compute_new_state_border (i, j);     
    }
  }
  return change;
}

static int do_tile_border (int x, int y, int width, int height, int who)
{
  int r;

  monitoring_start_tile (who);

  r = do_tile_reg_border (x, y, width, height);

  monitoring_end_tile (x, y, width, height, who);

  return r;
}

static int compute_new_state_nocheck (int y, int x)
{
  unsigned n      = 0;
  unsigned me     = cur_table (y, x) != 0;
  unsigned change = 0;

  for (int i = y - 1; i <= y + 1; i++)
    for (int j = x - 1; j <= x + 1; j++)
      n += cur_table (i, j);

  n = (n == 3 + me) | (n == 3);
  if (n != me)
    change |= 1;

  next_table (y, x) = n;
  

  return change;
}

static int do_tile_reg_nocheck (int x, int y, int width, int height)
{
  int change = 0;

  for (int i = y; i < y + height; i++)
    for (int j = x; j < x + width; j++)
      change |= compute_new_state_nocheck (i, j);

  return change;
}

static int do_tile_nocheck (int x, int y, int width, int height, int who)
{
  int r;

  monitoring_start_tile (who);

  r = do_tile_reg_nocheck (x, y, width, height);

  monitoring_end_tile (x, y, width, height, who);

  return r;
}
//
//////////////////////// BitMap lazy Version ;
// ./run -k life -s 128 -ts 32 -v lazybtmp -m
// ./run -k life -a random -s 2048 -ts 32 -v lazybtmp -m
// ./run -k life -a otca_off -s 2196 -ts 32 -v lazybtmp -m
unsigned life_compute_lazybtmp (unsigned nb_iter){
  
  unsigned change = 0;
  unsigned res=0;

  // init section of the data structures 
  if(init){ 
    changeLock = (omp_lock_t *) malloc(sizeof(omp_lock_t));
    bitMapTls = initBtmptls(changeLock,curTable);
    init=false;
  }
  //main loop
  for(unsigned it=1; it<=nb_iter;it++){
    //printBitmaps(bitMapTls,curTable);
    #pragma omp parallel for collapse(2) schedule(dynamic)
    for(int j = 0; j< NB_TILES_Y;j++){
      for(int i = 0; i< NB_TILES_X;i++){
        if(*(bitMapTls+curTable*NB_TILES_TOT+(j*NB_TILES_X)+i)==1){
            unsigned x=i * TILE_W;
            unsigned y=j * TILE_H;
            unsigned tileChange = 0;

            // do_tile compute the inner pixels of the tile, not bothering
            // if we should compute bordering tiles in the next itteration
            //
            // btmp_do_tile is used for the outer pixels of the tile, adding for the next
            // itteration the borduring tile if the current pixel has changed state
            
            tileChange |= do_tile (x +1, y +1, TILE_W-2, TILE_H-2, omp_get_thread_num());//inner tile
            tileChange |= do_tile_border(x,y,TILE_W,1,omp_get_thread_num());//top
            tileChange |= do_tile_border(x,y+TILE_H-1,TILE_W,1,omp_get_thread_num());//bot
            tileChange |= do_tile_border(x,y+1,1,TILE_H-2,omp_get_thread_num());//left
            tileChange |= do_tile_border(x+TILE_W-1,y+1,1,TILE_H-2,omp_get_thread_num());//right
            
            //If the tile changed, we'll want to compute it in the next itter
            if(tileChange){
              addTaskBtmp(i,j,bitMapTls+nextTable*NB_TILES_TOT);
            }
            change |= tileChange;
          
        }
      }
    }
    //printBitmaps(bitMapTls,curTable);
    deleteBtmp(bitMapTls+curTable*NB_TILES_TOT);
    switcher = !switcher;
    swap_tables ();

    if (!change) { // we stop when all cells are stable
      res = it;
      printf("there's no future tasks\n");
      //break;
    }
  }
return res;
}

static int compute_new_state_vec (int y, int x)
{

  
  //__m256i maskToAply =  _mm256_set1_epi8(1);


  __m256i LtopVec = _mm256_loadu_si256((void*)(&cur_table(y-1,x-1)));
  __m256i LmidVec = _mm256_loadu_si256((void*)(&cur_table(y,x-1)));
  __m256i LbotVec = _mm256_loadu_si256((void*)(&cur_table(y+1,x-1)));

  __m256i topVec = _mm256_loadu_si256((void*)(&cur_table(y-1,x)));
  __m256i midVec = _mm256_loadu_si256((void*)(&cur_table(y,x)));
  __m256i botVec = _mm256_loadu_si256((void*)(&cur_table(y+1,x)));

  __m256i RtopVec = _mm256_loadu_si256((void*)(&cur_table(y-1,x+1)));
  __m256i RmidVec = _mm256_loadu_si256((void*)(&cur_table(y,x+1)));
  __m256i RbotVec = _mm256_loadu_si256((void*)(&cur_table(y+1,x+1)));

  __m256i mask1 = _mm256_set1_epi8(1);
  __m256i change = _mm256_set1_epi8(0);
  __m256i nVec = _mm256_set1_epi8(0);
  __m256i meVec = _mm256_loadu_si256((void*)(&cur_table(y,x)));
  meVec = _mm256_and_si256(meVec,mask1);

  __m256i totVec = _mm256_add_epi8(\
                        _mm256_add_epi8(\
                          topVec,\
                          botVec),\
                        midVec);
  __m256i LtotVec = _mm256_add_epi8(\
                        _mm256_add_epi8(\
                          LtopVec,\
                          LbotVec),\
                        LmidVec);
  __m256i RtotVec = _mm256_add_epi8(\
                        _mm256_add_epi8(\
                          RtopVec,\
                          RbotVec),\
                        RmidVec);
  
  nVec =_mm256_add_epi8(totVec,\
          _mm256_add_epi8(LtotVec,RtotVec));
          
  __m256i neq3 = _mm256_and_si256(_mm256_cmpeq_epi8(nVec,_mm256_set1_epi8(3)),mask1);
  __m256i meP3 = _mm256_add_epi8(meVec,_mm256_set1_epi8(3));
  __m256i neqMeP3 = _mm256_and_si256(_mm256_cmpeq_epi8(nVec,meP3),mask1);

  nVec =  _mm256_or_si256(neqMeP3,neq3);
  change = _mm256_xor_si256(nVec,meVec);
  nVec = _mm256_and_si256(nVec,mask1);

  _mm256_storeu_si256((void*)&(next_table(y,x)),nVec);
  
  bool vecChange = ! _mm256_testz_si256(_mm256_or_si256(change,_mm256_setzero_si256()), _mm256_set1_epi8(1));
      //printf("change : %d\n",vecChange);
  // if(setback!=0)
  //   exit(0);
  return vecChange;
}

static int do_tile_reg_vec (int x, int y, int width, int height)
{
  int change = 0;
  
  if(width<(VEC_SIZE_CHAR))
    change |= do_tile_nocheck (x, y, width, height, omp_get_thread_num());
  else
    for (int i = y; i < y + height; i++)
      for (int j = x; j < x + width; j+=VEC_SIZE_CHAR){
        int setback = ((j+VEC_SIZE_CHAR)>=width+x)*(32-((x+width)-j)) ;
        //printf("setback %d \n tile %d * %d ",setback,width,height);
      
        change |= compute_new_state_vec (i, j-setback);
      }

  return change;
}

static int do_tile_vec (int x, int y, int width, int height, int who)
{
  int r;

  monitoring_start_tile (who);

  r = do_tile_reg_vec (x, y, width, height);

  monitoring_end_tile (x, y, width, height, who);

  return r;
}

/**
 * Calls the best kernel with the best parameters according
 * to the tile position amongst others
 * 
 * example, if the tile is surrounded by other tiles, it wont 
 * verify the tiles borders
 * 
 */
bool tileLauncher (unsigned i, unsigned j){
  bool tileChange = false;
  unsigned x=i * TILE_W;
  unsigned y=j * TILE_H;
  tileChange = do_tile_border(x,y,TILE_W,1,omp_get_thread_num());//top
  tileChange |= do_tile_border(x,y+1,1,TILE_H-2,omp_get_thread_num());//left
  tileChange |= do_tile_vec (x+1, y+1 , TILE_W-2, TILE_H-2, omp_get_thread_num());//inner tile
  tileChange |= do_tile_border(x+TILE_W-1,y+1,1,TILE_H-2,omp_get_thread_num());//right
  tileChange |= do_tile_border(x,y+TILE_H-1,TILE_W,1,omp_get_thread_num());//bot
  
  return tileChange;
}

//////////////////////// BitMap2 lazy vectorial Version ;
// ./run -k life -s 2048 -ts 64 -v lazybtmpvec -m
// ./run -k life -a random -s 2048 -ts 32 -v lazybtmpvec -m
// ./run -k life -a otca_off -s 2196 -ts 61 -v lazybtmpvec -m
unsigned life_compute_lazybtmpvec (unsigned nb_iter){
  unsigned change = 0;
  unsigned res=0;

  if(init){ 
    changeLock = (omp_lock_t *) malloc(sizeof(omp_lock_t));
    bitMapTls = initBtmptls(changeLock,curTable);
    init=false;
  }

  for(unsigned it=1; it<=nb_iter;it++){

    #pragma omp parallel for collapse(2) schedule(dynamic)
    for(int i = 0; i< NB_TILES_X;i++){
      for(int j = 0; j< NB_TILES_Y;j++){
        if(*(bitMapTls+curTable*NB_TILES_TOT+(j*NB_TILES_X)+i)){
            unsigned tileChange = false;
            tileChange = tileLauncher(i,j);
            if(tileChange){
              addTaskBtmp(i,j,bitMapTls+nextTable*NB_TILES_TOT);
            }
            change |= tileChange;    
        }
      }
    }
    deleteBtmp(bitMapTls+curTable*NB_TILES_TOT);
    switcher = !switcher;
    swap_tables ();
    if (!change) { // we stop when all cells are stable
      res = it;
      printf("there's no future tasks\n");
      //break;
    }
  }
return res;
}

//////////////////////////////// Tiled omp version
// ./run -k random -s 1024 -ts 128 -i 100 -n -v omp
unsigned life_compute_ompvec (unsigned nb_iter)
{

  unsigned res = 0;

  for (unsigned it = 1; it <= nb_iter; it++) {
    unsigned change = 0;
    #pragma omp parallel for collapse(2) schedule (dynamic)
    for (int y = 0; y < DIM; y += TILE_H)
      for (int x = 0; x < DIM; x += TILE_W){
        unsigned tilepos =tilePosition(x/TILE_W,y/TILE_H);
        unsigned who = 0;//omp_get_thread_num();
        switch(tilepos){
          case top : 
            change |= do_tile(x, y, TILE_W, 1, who);//top
            change |= do_tile_vec (x, y+1, TILE_W, TILE_H-1, who);
          ;break;
          case topleft : 
            change |= do_tile(x, y, TILE_W, 1, who);//top
            change |= do_tile(x, y-1, 1, TILE_H-1, who);//left
            change |= do_tile_vec (x+1, y+1, TILE_W-1, TILE_H-1, who);
          ;break;
          case left : 
            change |= do_tile(x, y, 1, TILE_H, who);//left
            change |= do_tile_vec (x+1, y, TILE_W-1, TILE_H, who);
          ;break;
          case botleft : 
            change |= do_tile(x, y+TILE_H-1, TILE_W, 1, who);//bot
            change |= do_tile(x, y, 1, TILE_H-1, who);//left
            change |= do_tile_vec (x+1, y, TILE_W-1, TILE_H-1, who);
          ;break;
          case bot : 
            change |= do_tile(x, y+TILE_H-1, TILE_W, 1, who);//bot
            change |= do_tile_vec (x, y, TILE_W, TILE_H-1, who);
          ;break;
          case botright : 
            change |= do_tile(x, y+TILE_H-1, TILE_W, 1, who);//bot
            change |= do_tile(x+TILE_W-1, y, 1, TILE_H-1, who);//right
            change |= do_tile_vec (x, y, TILE_W-1, TILE_H-1, who);
          ;break;
          case right : 
            change |= do_tile(x+TILE_W-1, y, 1, TILE_H, who);//right
            change |= do_tile_vec (x, y, TILE_W-1, TILE_H, who);
          ;break;
          case topright : 
            change |= do_tile(x, y, TILE_W, 1, who);//top
            change |= do_tile(x+TILE_W-1, y-1, 1, TILE_H-1, who);//right
            change |= do_tile_vec (x, y+1, TILE_W-1, TILE_H-1, who);
          ;break;

          default: change |= do_tile_vec (x, y, TILE_W, TILE_H, who);
          break;
        }

        
      }
    swap_tables ();
    
    if (!change) { // we stop when all cells are stable
      res = it;
      break;
    }
  }
  return res;
}

///////////////////////////// Initial configs

void life_draw_guns (void);

static inline void set_cell (int y, int x)
{
  cur_table (y, x) = 1;
  if (opencl_used)
    cur_img (y, x) = 1;
}

static inline int get_cell (int y, int x)
{
  return cur_table (y, x);
}

static void inline life_rle_parse (char *filename, int x, int y,
                                   int orientation)
{
  rle_lexer_parse (filename, x, y, set_cell, orientation);
}

static void inline life_rle_generate (char *filename, int x, int y, int width,
                                      int height)
{
  rle_generate (x, y, width, height, get_cell, filename);
}

void life_draw (char *param)
{
  if (param && (access (param, R_OK) != -1)) {
    // The parameter is a filename, so we guess it's a RLE-encoded file
    life_rle_parse (param, 1, 1, RLE_ORIENTATION_NORMAL);
  } else
    // Call function ${kernel}_draw_${param}, or default function (second
    // parameter) if symbol not found
    hooks_draw_helper (param, life_draw_guns);
}

static void otca_autoswitch (char *name, int x, int y)
{
  life_rle_parse (name, x, y, RLE_ORIENTATION_NORMAL);
  life_rle_parse ("data/rle/autoswitch-ctrl.rle", x + 123, y + 1396,
                  RLE_ORIENTATION_NORMAL);
}

static void otca_life (char *name, int x, int y)
{
  life_rle_parse (name, x, y, RLE_ORIENTATION_NORMAL);
  life_rle_parse ("data/rle/b3-s23-ctrl.rle", x + 123, y + 1396,
                  RLE_ORIENTATION_NORMAL);
}

static void at_the_four_corners (char *filename, int distance)
{
  life_rle_parse (filename, distance, distance, RLE_ORIENTATION_NORMAL);
  life_rle_parse (filename, distance, distance, RLE_ORIENTATION_HINVERT);
  life_rle_parse (filename, distance, distance, RLE_ORIENTATION_VINVERT);
  life_rle_parse (filename, distance, distance,
                  RLE_ORIENTATION_HINVERT | RLE_ORIENTATION_VINVERT);
}

// Suggested cmdline: ./run -k life -s 2176 -a otca_off -ts 64 -r 10 -si
void life_draw_otca_off (void)
{
  if (DIM < 2176)
    exit_with_error ("DIM should be at least %d", 2176);

  otca_autoswitch ("data/rle/otca-off.rle", 1, 1);
}

// Suggested cmdline: ./run -k life -s 2176 -a otca_on -ts 64 -r 10 -si
void life_draw_otca_on (void)
{
  if (DIM < 2176)
    exit_with_error ("DIM should be at least %d", 2176);

  otca_autoswitch ("data/rle/otca-on.rle", 1, 1);
}

// Suggested cmdline: ./run -k life -s 6208 -a meta3x3 -ts 64 -r 50 -si
void life_draw_meta3x3 (void)
{
  if (DIM < 6208)
    exit_with_error ("DIM should be at least %d", 6208);

  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++)
      otca_life (j == 1 ? "data/rle/otca-on.rle" : "data/rle/otca-off.rle",
                 1 + j * (2058 - 10), 1 + i * (2058 - 10));
}

// Suggested cmdline: ./run -k life -a bugs -ts 64
void life_draw_bugs (void)
{
  for (int y = 16; y < DIM / 2; y += 32) {
    life_rle_parse ("data/rle/tagalong.rle", y + 1, y + 8,
                    RLE_ORIENTATION_NORMAL);
    life_rle_parse ("data/rle/tagalong.rle", y + 1, (DIM - 32 - y) + 8,
                    RLE_ORIENTATION_NORMAL);
  }
}

// Suggested cmdline: ./run -k life -v omp -a ship -s 512 -m -ts 16
void life_draw_ship (void)
{
  for (int y = 16; y < DIM / 2; y += 32) {
    life_rle_parse ("data/rle/tagalong.rle", y + 1, y + 8,
                    RLE_ORIENTATION_NORMAL);
    life_rle_parse ("data/rle/tagalong.rle", y + 1, (DIM - 32 - y) + 8,
                    RLE_ORIENTATION_NORMAL);
  }

  for (int y = 43; y < DIM - 134; y += 148) {
    life_rle_parse ("data/rle/greyship.rle", DIM - 100, y,
                    RLE_ORIENTATION_NORMAL);
  }
}

void life_draw_stable (void)
{
  for (int i = 1; i < DIM - 2; i += 4)
    for (int j = 1; j < DIM - 2; j += 4) {
      set_cell (i, j);
      set_cell (i, j + 1);
      set_cell (i + 1, j);
      set_cell (i + 1, j + 1);
    }
}

void life_draw_guns (void)
{
  at_the_four_corners ("data/rle/gun.rle", 1);
}

void life_draw_random (void)
{
  for (int i = 1; i < DIM - 1; i++)
    for (int j = 1; j < DIM - 1; j++)
      if (random () & 1)
        set_cell (i, j);
}

// Suggested cmdline: ./run -k life -a clown -s 256 -i 110
void life_draw_clown (void)
{
  life_rle_parse ("data/rle/clown-seed.rle", DIM / 2, DIM / 2,
                  RLE_ORIENTATION_NORMAL);
}

void life_draw_diehard (void)
{
  life_rle_parse ("data/rle/diehard.rle", DIM / 2, DIM / 2,
                  RLE_ORIENTATION_NORMAL);
}
