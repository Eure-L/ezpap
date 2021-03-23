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
#define NB_FAKE_X (NB_TILES_X + 2)
#define NB_FAKE_Y (NB_TILES_Y + 2)
#define NB_FAKE_TILES ((NB_FAKE_X)*(NB_FAKE_Y))



static unsigned color = 0xFFFF00FF; // Living cells have the yellow color

typedef char cell_t;

static cell_t *restrict _table = NULL, *restrict _alternate_table = NULL;

bool switcher = 1;

#define curTable switcher
#define nextTable !switcher

char * bitMapTls; // Two Bit maps represented in a vector
                  // Each bits representing a tile to compute
__m256i zero;
static inline cell_t *table_cell (cell_t *restrict i, int y, int x)
{
  //added empty border for optimized structure for AVX usage
  return i + (y+1) * (DIM+VEC_SIZE_CHAR*2) + (x + VEC_SIZE_CHAR);
}

static inline char *table_map (char * i, int x, int y)
{
  return i + (y+1) * NB_FAKE_X + (x+1);
}


// This kernel does not directly work on cur_img/next_img.
// Instead, we use 2D arrays of boolean values, not colors
#define cur_table(y, x) (*table_cell (_table, (y), (x)))
#define next_table(y, x) (*table_cell (_alternate_table, (y), (x)))
#define next_tableAddr(y, x) (table_cell (_alternate_table, (y), (x)))

#define cur_map(x, y) (*table_map ((bitMapTls+(curTable*NB_FAKE_TILES)), (x), (y)))
#define cur_mapAddr(x, y) (table_map ((bitMapTls+(curTable*NB_FAKE_TILES)), (x), (y)))
#define next_map(x, y) (*table_map ((bitMapTls+(nextTable*NB_FAKE_TILES)), (x), (y)))
#define next_mapAddr(x, y) (table_map ((bitMapTls+(nextTable*NB_FAKE_TILES)), (x), (y)))

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
static inline bool isOnLeft(int x,int y){
  return x > 0 && x%TILE_W == 0;
}
static inline bool isOnRight(int x,int y){
  return (x<DIM-1 && (x%TILE_W == TILE_W-1));
}
static inline bool isOnTop(int x,int y){
  return (y>0 && y%TILE_H==0);
}
static inline bool isOnBottom(int x,int y){
  return (y<DIM-1 && (y%TILE_H == TILE_H-1));
}
//
static inline unsigned isTileOnLeft(int i,int j){
  return i == 0;
}
static inline unsigned isTileOnRight(int i,int j){
  return i == NB_TILES_X-1;
}
static inline unsigned isTileOnTop(int i,int j){
  return j == 0;
}
static inline unsigned isTileOnBottom(int i,int j){
  return j == NB_TILES_Y-1;
}
////



// This function is called whenever the graphical window needs to be refreshed
void life_refresh_img (void)
{
  for (int i = 0; i < DIM; i++)
    for (int j = 0; j < DIM; j++)
      cur_img (i, j) = cur_table (i, j) * color;
}


char * initBtmptls(void){
  
  char * map = (char *) malloc ((2 * (NB_FAKE_TILES) * sizeof(char)));
  if(map == NULL){
    printf("map pointer NULL\n");
    exit(EXIT_FAILURE);
  }       

  //int idMap;
  for (int i=0;i<NB_FAKE_TILES;i++){
    *(map+i)=0;
  }
  for (int i=0;i<NB_FAKE_TILES;i++){
    *(map+NB_FAKE_TILES+i)=1;
  }
    //printf("init a : %d \n",*(map+(NB_TILES_TOT)+i));
  
  return map;
}

void printBitmaps(void){

  printf("Current bitmap : \n");
  for(int i = 0; i<NB_TILES_X; i++){
    for(int j = 0; j<NB_TILES_Y; j++){
      printf(" %d ", cur_map(i,j));
    }
    printf("\n");
  }
  printf("\n Next bitmap : \n");
  for(int i = 0; i<NB_TILES_X; i++){
    for(int j = 0; j<NB_TILES_Y; j++){
      printf(" %d ", next_map(i,j));
    }
    printf("\n");
  }
}

void deleteCurrentBtmp(){ 
  #pragma omp parallel for schedule(dynamic)
  for(int j = 0;j<NB_TILES_Y; j++){
    for(int i = 0; i<NB_TILES_X; i+=VEC_SIZE_CHAR){
      _mm256_storeu_si256((void*)cur_mapAddr(i,j),zero);
    }
  }
}

unsigned tilePosition(int i, int j){
   return top * isTileOnTop(i,j) + bot * isTileOnBottom(i,j) + 
          left * isTileOnLeft(i,j) + right * isTileOnRight(i,j);
}

void prntAVXi( __m256i vec,char * name){
  int byteSize = sizeof(char);
  const size_t n = sizeof(__m256i) / byteSize;
  char buffer[n];
  _mm256_storeu_si256((void*)buffer, vec);
  int i = 0;
  printf("%s : ",name);
  for (; i < 32 ; i++){
      //if(buffer[i]!=0)
         
      printf("%u.",buffer[i]);
      
  }
  printf(" ~ i : %d\n",i);
}

static inline bool hasNeighbourChanged(unsigned i,unsigned j){
  //printf("isok\n");
  // #define cur_map(y, x) (*table_map ((bitMapTls+(curTable*NB_FAKE_TILES)), (y), (x)))
  
  bool nChanged = cur_map(i,j)|cur_map(i-1,j)|cur_map(i+1,j)|cur_map(i,j-1)|cur_map(i,j+1)
  |cur_map(i-1,j-1)|cur_map(i+1,j+1)|cur_map(i+1,j-1)|cur_map(i-1,j+1);

  return  nChanged;
}

void life_init (void)
{
  // life_init may be (indirectly) called several times so we check if data were
  // already allocated
  if (_table == NULL) {
    const unsigned size = (DIM+VEC_SIZE_CHAR*2) * (DIM+VEC_SIZE_CHAR*2) * sizeof (cell_t);
    PRINT_DEBUG ('u', "Memory footprint = 2 x %d bytes\n", size);

    _table = mmap (NULL, size, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0 );

    _alternate_table = mmap (NULL, size, PROT_READ | PROT_WRITE,
                             MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  }
  if(bitMapTls==NULL)
    bitMapTls = initBtmptls(); 
  zero = _mm256_setzero_si256(); 
}

void life_finalize (void)
{
  // printf("finalize\n");
  const unsigned size = (DIM+VEC_SIZE_CHAR*2) * (DIM+VEC_SIZE_CHAR*2)  * sizeof (cell_t);

  munmap (_table, size);
  munmap (_alternate_table, size);
  
  if(bitMapTls!=NULL)
    free(bitMapTls);
}

static inline void swap_tables (void)
{
  cell_t *tmp = _table;
  switcher = !switcher;
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
        *next_mapAddr((x-TILE_W)/TILE_W,(y/TILE_H))=1;    
        if(istop)
          *next_mapAddr((x-TILE_W)/TILE_W,(y-TILE_H)/TILE_H)=1;
        else if(isbottom)
          *next_mapAddr((x-TILE_W)/TILE_W,(y+1)/TILE_H)=1;
      }
      else if (isright){
        *next_mapAddr((x+1)/TILE_W,(y/TILE_H))=1;  
        if(istop)
          *next_mapAddr((x+1)/TILE_W,(y-TILE_H)/TILE_H)=1;
        else if(isbottom)
          *next_mapAddr((x+1)/TILE_W,(y+1)/TILE_H)=1;   
      }
      if(istop){
        *next_mapAddr((x/TILE_W),(y-TILE_H)/TILE_H)=1;   
      }
      else if (isbottom){
        *next_mapAddr((x/TILE_W),(y+1)/TILE_H)=1;
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

  //main loop
  for(unsigned it=1; it<=nb_iter;it++){
    //printBitmaps();
    #pragma omp parallel for collapse(2) schedule(dynamic)
    for(int j = 0; j< NB_TILES_Y;j++){
      for(int i = 0; i< NB_TILES_X;i++){
        //if(*(bitMapTls+curTable*NB_FAKE_TILES+(j*NB_TILES_X)+i)==1){
        //if(hasNeighbourChanged(i,j)){
          if(cur_map(i,j)){
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
              *next_mapAddr(i,j)=1;
            }
            change |= tileChange; 
        }
      }
    }
    //printBitmaps();
    deleteCurrentBtmp();
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

void printVecLanes(__m256i * vecLst, int topL, int midL, int botL,char * str){
  printf("%s",str);
  prntAVXi(vecLst[topL],"top");
  prntAVXi(vecLst[midL],"mid");
  prntAVXi(vecLst[botL],"bot");
}

/////////////////////////////////////////// vectorial version

//must be called on tiles of width's sizes multiple of 32
static int do_tile_reg_vec (int x, int y, int width, int height)
{
  unsigned  tileChange = 0;
 
  __m256i vecTabLeft[3];
  __m256i vecTabMid[3];
  __m256i vecTabRight[3];
  
  __m256i mask1 = _mm256_set1_epi8(1);
  __m256i change = _mm256_set1_epi8(0);
  __m256i nVec = _mm256_set1_epi8(0);

  unsigned cnt = 0;
  
  #define toplane  ((cnt)%3)
  #define midlane  ((cnt+1)%3)
  #define botlane ((cnt+2)%3)
  
  unsigned i = x;
  

  for ( i = x; i < x + width; i+=VEC_SIZE_CHAR){
    unsigned j = y;
    cnt = 0; //counts lines

    vecTabLeft[toplane] = _mm256_loadu_si256((void*)(&cur_table(j-1,i-1)));
    vecTabLeft[midlane] = _mm256_loadu_si256((void*)(&cur_table(j,i-1)));
    vecTabLeft[botlane] = _mm256_loadu_si256((void*)(&cur_table(j+1,i-1)));

    vecTabRight[toplane] = _mm256_loadu_si256((void*)(&cur_table(j-1,i+1)));
    vecTabRight[midlane] = _mm256_loadu_si256((void*)(&cur_table(j,i+1)));
    vecTabRight[botlane] = _mm256_loadu_si256((void*)(&cur_table(j+1,i+1)));

    vecTabMid[toplane] = _mm256_load_si256((void*)(&cur_table(j-1,i)));
    vecTabMid[midlane] = _mm256_load_si256((void*)(&cur_table(j,i)));
    vecTabMid[botlane] = _mm256_load_si256((void*)(&cur_table(j+1,i)));
    
    for ( j = y; j < y + height; j++){
      //printf("       %d / %d\n",j-y,height);
      // printf("\n\n\n=============New Line===========\n");
      // printf(" %d - %d / %d\n",i,i+height,y+height);
      // printVecLanes(vecTabMid,toplane,midlane,botlane,"MIDDLE\n");
      //printVecLanes(vecTabLeft,toplane,midlane,botlane,"LEFT\n");
      //printVecLanes(vecTabRight,toplane,midlane,botlane,"RIGHT\n");

      __m256i MtotVec = _mm256_add_epi8(\
                            _mm256_add_epi8(\
                              vecTabMid[toplane],\
                              vecTabMid[midlane]),\
                            vecTabMid[botlane]);
      __m256i LtotVec = _mm256_add_epi8(\
                            _mm256_add_epi8(\
                              vecTabLeft[toplane],\
                              vecTabLeft[midlane]),\
                            vecTabLeft[botlane]);
      __m256i RtotVec = _mm256_add_epi8(\
                            _mm256_add_epi8(\
                              vecTabRight[toplane],\
                              vecTabRight[midlane]),\
                            vecTabRight[botlane]);

      nVec =_mm256_add_epi8(MtotVec,\
              _mm256_add_epi8(LtotVec,RtotVec));
      
      // prntAVXi(nVec,"NVc");

      __m256i neq3 = _mm256_and_si256(_mm256_cmpeq_epi8(nVec,_mm256_set1_epi8(3)),mask1);
      __m256i meP3 = _mm256_add_epi8(vecTabMid[midlane],_mm256_set1_epi8(3));
      __m256i neqMeP3 = _mm256_and_si256(_mm256_cmpeq_epi8(nVec,meP3),mask1);

      nVec =  _mm256_or_si256(neqMeP3,neq3);
      change = _mm256_xor_si256(nVec,vecTabMid[midlane]);
      nVec = _mm256_and_si256(nVec,mask1);

      // printf("\nWho must live\n");
      // prntAVXi(nVec,"Nvc");

      _mm256_storeu_si256((void*)(next_tableAddr(j,i)),nVec);
      
      bool vecChange = ! _mm256_testz_si256(_mm256_or_si256(change,_mm256_setzero_si256()), _mm256_set1_epi8(1));
      tileChange |= vecChange;

      //  printf("\n----rollout----\n");
      // printVecLanes(vecTabMid,toplane,midlane,botlane,"MIDDLE\n");
      // printVecLanes(vecTabLeft,toplane,midlane,botlane,"LEFT\n");
      // printVecLanes(vecTabRight,toplane,midlane,botlane,"RIGHT\n");

      // Rolling the roles
      vecTabRight[toplane]=_mm256_loadu_si256((void*)(&cur_table(j+2,i+1)));
      vecTabLeft[toplane]=_mm256_loadu_si256((void*)(&cur_table(j+2,i-1)));
      vecTabMid[toplane]=_mm256_load_si256((void*)(&cur_table(j+2,i)));
      cnt++;
      // printf("\n---rollout done\n");
      // printVecLanes(vecTabMid,toplane,midlane,botlane,"MIDDLE\n");
      // printVecLanes(vecTabLeft,toplane,midlane,botlane,"LEFT\n");
      // printVecLanes(vecTabRight,toplane,midlane,botlane,"RIGHT\n");

      // if(!_mm256_testz_si256(_mm256_xor_si256(nVec,_mm256_setzero_si256()), _mm256_set1_epi32(1)))
      //   exit(0);
    }
  }
  // if(tileChange)
  //   exit(0);

  return tileChange;
}

static int do_tile_vec (int x, int y, int width, int height, int who)
{
  int r;

  monitoring_start_tile (who);

  r = do_tile_reg_vec (x, y, width, height);

  monitoring_end_tile (x, y, width, height, who);

  return r;
}

static bool hasAnyTileChanged(void){
  bool change = false;
  #pragma omp parallel for schedule (static) shared(change)
  for(int j = 0 ; j < NB_TILES_Y; j++){
    if(change){
        continue;
      }
    for(int i = 0 ; i < NB_TILES_X; i++){
      if(change){
        continue;
      }
      if(cur_map(i,j))
        change = true;
    }
  }
  return change;
}


//////////////////////// BitMap2 lazy vectorial Version ;
// ./run -k life -s 2048 -ts 64 -v lazybtmpvec -m
// ./run -k life -a random -s 2048 -ts 32 -v lazybtmpvec -m
// ./run -k life -a otca_off -s 2196 -ts 61 -v lazybtmpvec -m
unsigned life_compute_lazybtmpvec (unsigned nb_iter){
  unsigned change = 0;
  unsigned res=0;

  for(unsigned it=1; it<=nb_iter;it++){
    #pragma omp parallel for collapse(2) schedule(dynamic)
    for(int j = 0; j< NB_TILES_Y;j++){ 
      for(int i = 0; i< NB_TILES_X;i++){
        if(hasNeighbourChanged(i,j)){
          unsigned x=i * TILE_W;
          unsigned y=j * TILE_H;
          unsigned who = omp_get_thread_num();
          #if 0
          unsigned tileChange = false;
          tileChange = do_tile_vec(x, y , TILE_W, TILE_H, who);
          *next_mapAddr(i,j)=tileChange;
          change |= tileChange;
          #else
            *next_mapAddr(i,j)=do_tile_vec(x, y , TILE_W, TILE_H, who);

          #endif    
        }
      } 
    }
    #if 0
    #else
    change = hasAnyTileChanged();
    #endif
    deleteCurrentBtmp();
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
    #pragma omp parallel for schedule (dynamic)
    for (int y = 0; y < DIM; y+=TILE_H)
      for (int x = 0; x < DIM; x+=TILE_W){
        change |=  do_tile_vec(x, y , TILE_W, TILE_H, omp_get_thread_num());   
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
