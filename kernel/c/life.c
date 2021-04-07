#include "easypap.h"
#include "rle_lexer.h"
#include "arch_flags.h"

#include <avxintrin.h>
#include <omp.h>
#include <stdbool.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

typedef char cell_t;
static unsigned color = 0xFFFF00FF; // Living cells have the yellow color

static cell_t *restrict _table = NULL, *restrict _alternate_table = NULL;
char * bitMapTls; // Two Bit maps represented in a vector
                  // Each bits representing a tile to compute
bool switcher = 1;
__m256i mask1;
__m256i zero;
bool one = !0;
char threadChange[256];
char threadBuffer[256];

#define AVXBITS 256
#define VEC_SIZE (AVXBITS/(sizeof(cell_t)*8))


#define NB_TILES_TOT (NB_TILES_X*NB_TILES_Y)
#define NB_FAKE_X (NB_TILES_X + 2)
#define NB_FAKE_Y (NB_TILES_Y + 2)
#define NB_FAKE_TILES ((NB_FAKE_X)*(NB_FAKE_Y))
#define curTable switcher
#define nextTable !switcher

unsigned bits;
bool ENABLE_BITCELL = 0;
unsigned SIZEX ;
unsigned SIZEY ;

#define _table_SIZE (SIZEX*SIZEY*sizeof(cell_t))
#define DIMTOT (SIZEX*SIZEY)

//returns the word containing the cell // must do a bit shift on the ptr
static inline cell_t *table_cell_column (cell_t *restrict i, int y, int x)
{ 
  return i + (((y+1)/8)+1) * (SIZEX) + (x + 1);
}

static inline cell_t *table_cell_row (cell_t *restrict i, int y, int x)
{ 
  return i + (y+1) * (SIZEX) + (((x+1)/8) + 1);
}

static inline cell_t *table_cell (cell_t *restrict i, int y, int x)
{
  return i + (y+1) * (DIM+2) + (x + 1);
}

static inline char *table_map (char * i, int x, int y)
{
  return i + (y+1) * NB_FAKE_X + (x+1);
}

static inline char *fake_map (char * i, int x, int y)
{
  return i + (y) * NB_FAKE_X + (x);
}


// This kernel does not directly work on cur_img/next_img.
// Instead, we use 2D arrays of boolean values, not colors
#define cur_table(y, x) (*table_cell (_table, (y), (x)))
#define next_table(y, x) (*table_cell (_alternate_table, (y), (x)))

#define cur_table_bits(y, x) ((*table_cell_column (_table, (y), (x))))
#define next_table_bits(y, x) (*table_cell_column (_alternate_table, (y), (x)))

#define cur_map(x, y) (*table_map ((bitMapTls+(curTable*NB_FAKE_TILES)), (x), (y)))
#define cur_fmap(x, y) (*fake_map ((bitMapTls+(curTable*NB_FAKE_TILES)), (x), (y)))
#define next_map(x, y) (*table_map ((bitMapTls+(nextTable*NB_FAKE_TILES)), (x), (y)))
#define next_fmap(x, y) (*fable_map ((bitMapTls+(nextTable*NB_FAKE_TILES)), (x), (y)))

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



static inline void setnextBitCell(int i, int j, unsigned val){
  next_table_bits(j,i) = (next_table_bits(j,i) & ~(0x01<<((j)%8))) | (val<<((j)%8));
}

static inline void setcurBitCell(int i, int j, unsigned val){
  // printB(cur_table_bits(j,i));
  // printf("shift de : %d\n",j%8);
  // printB((cur_table_bits(j,i) & ~(0x01<<((j)%8))) | (val<<((j)%8)));
  // printf("\n\n");
  cur_table_bits(j,i) = (cur_table_bits(j,i) & ~(0x01<<((j)%8))) | (val<<((j)%8));
}

//
static inline cell_t getBitCell(int i, int j){
  // printB(cur_table_bits(j,i));
  // printf("on veut le : %d\n",j%8);
  // printf("%d\n",cur_table_bits(j,i)>>((j)%8)&(0x01));
  // printf("\n\n");
  return (cur_table_bits(j,i)>>((j)%8)&(0x01));
}

//
void printB(char mot){
  for(int i=0;i<8;i++){
    printf("%d ",(mot>>i)&0x01);
  }
  //exit(0);
}

void printTable(cell_t * table){
  printf("DIM %d\n",DIM);
  printf("SIZEX %d SIZEY %d\n",SIZEX,SIZEY);
  for(int i =0; i< SIZEY;i++){
    for(int j=0; j<SIZEX;j++){
      if(ENABLE_BITCELL){
        if(i%8==0){
          //printf("%d ",*(table+j*SIZEX+i));
          printB((char)getBitCell(j,i));
        }
      }
      else{
        printf("%d ",*(table+i*SIZEX+j));
      }
    }
    printf("\n");
  }
  printf("\n------------\n\n");
}

void clear_table(cell_t * table){
  for(int i =0; i<DIMTOT;i++)
    *(table+i)=0;
}


// This function is called whenever the graphical window needs to be refreshed
void life_refresh_img (void)
{
  for (int i = 0; i < DIM; i++)
    for (int j = 0; j < DIM; j++)
        cur_img (i, j) = cur_table (i, j) * color;
}

void life_refresh_img_bits (void)
{
  for (int i = 0; i < DIM; i++){
    for (int j = 0; j < DIM; j++){
        cur_img (i, j) = (getBitCell (j, i) )* color;
    }
  }
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
  #if 0
  #pragma omp parallel for schedule(dynamic)
  for(int j = 0;j<NB_TILES_Y; j++){
    for(int i = 0; i<NB_TILES_X; i+=VEC_SIZE){
      _mm256_storeu_si256((void*)cur_mapAddr(i,j),zero);
    }
  }
  #else
  #pragma omp parallel for schedule(dynamic)
  for(int j = 0;j<NB_FAKE_Y; j++){
    for(int i = 0; i<NB_FAKE_X; i++){
      cur_fmap(i,j)=0;
    }
  }
  #endif
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
  
  return cur_map(i,j)|cur_map(i-1,j)|cur_map(i+1,j)|cur_map(i,j-1)|cur_map(i,j+1)
  |cur_map(i-1,j-1)|cur_map(i+1,j+1)|cur_map(i+1,j-1)|cur_map(i-1,j+1);

    
}

void life_init (void)
{
    printf("init normal\n");

  bits = sizeof(cell_t)*8;
  SIZEX =(DIM+(2));
  SIZEY =(DIM+16);
  if (_table == NULL) {
    const unsigned size = _table_SIZE;
    PRINT_DEBUG ('u', "Memory footprint = 2 x %d bytes\n", size);

    _table = mmap (NULL, size, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0 );

    _alternate_table = mmap (NULL, size, PROT_READ | PROT_WRITE,
                             MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  }
  if(bitMapTls==NULL)
    bitMapTls = initBtmptls(); 
  zero = _mm256_setzero_si256(); 
  mask1 = _mm256_set1_epi8(1);
}

void life_finalize (void)
{
  // printf("finalize\n");
  const unsigned size = _table_SIZE;

  munmap (_table, size);
  munmap (_alternate_table, size);
  
  if(bitMapTls!=NULL)
    free(bitMapTls);
}

void life_init_ocl(void)
{
  SIZEX =(DIM+(2));
  SIZEY =(DIM+2);
  if (_table == NULL) {
    const unsigned size = _table_SIZE;
    PRINT_DEBUG ('u', "Memory footprint = 2 x %d bytes\n", size);

    _table = mmap (NULL, size, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0 );

    _alternate_table = mmap (NULL, size, PROT_READ | PROT_WRITE,
                             MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  }
  if(bitMapTls==NULL)
    bitMapTls = initBtmptls(); 
  zero = _mm256_setzero_si256(); 
  mask1 = _mm256_set1_epi8(1);
  
}

void life_init_ocl2(void){
  life_init_ocl();
}

void life_init_bits(void){
  printf("init bits\n");
  ENABLE_BITCELL = 1;
  bits = sizeof(cell_t)*8;
  SIZEX =(DIM+(2));
  SIZEY =(DIM/bits)+2;
  if (_table == NULL) {
    const unsigned size = _table_SIZE;
    PRINT_DEBUG ('u', "Memory footprint = 2 x %d bytes\n", size);

    _table = mmap (NULL, size, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0 );

    _alternate_table = mmap (NULL, size, PROT_READ | PROT_WRITE,
                             MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  }
  if(bitMapTls==NULL)
    bitMapTls = initBtmptls(); 
  zero = _mm256_setzero_si256(); 
  mask1 = _mm256_set1_epi8(1);
  clear_table(_table);

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

static int compute_byte (int y, int x)
{
  unsigned n      = 0;
  unsigned me     = getBitCell(x,y);
  unsigned change = 0;
  
  for (int i = y - 1; i <= y + 1; i++)
    for (int j = x - 1; j <= x + 1; j++){
      n += getBitCell(j,i);
    }
  n = (n == 3 + me) | (n == 3);
  if (n != me)
    change |= 1;
  setnextBitCell(x,y,n);
  return 1;
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

void printVecLanes(__m256i * vecLst, int topL, int midL, int botL,char * str){
  printf("%s",str);
  prntAVXi(vecLst[topL],"top");
  prntAVXi(vecLst[midL],"mid");
  prntAVXi(vecLst[botL],"bot");
}

/////////////////////////////////////////// vectorial version

static int do_tile_reg_vec_bits (int x, int y, int width, int height)
{
  unsigned  tileChange = 0;
 
  __m256i vecTabLeft[3];
  __m256i vecTabMid[3];
  __m256i vecTabRight[3];
  __m256i MtotVec;
  __m256i LtotVec;
  __m256i RtotVec;

  __m256i neq3;
  __m256i meP3;
  __m256i neqMeP3;

  __m256i change;
  __m256i nVec;

  unsigned cnt = 0;
  
  #define toplane  ((cnt)%3)
  #define midlane  ((cnt+1)%3)
  #define botlane ((cnt+2)%3)
  
  unsigned i = x;
  

  for ( i = x; i < x + width; i+=VEC_SIZE){
    unsigned j = y;
    cnt = 0; //counts lines

    vecTabLeft[toplane] = _mm256_loadu_si256((void*)(&cur_table(j-1,i-1)));
    vecTabLeft[midlane] = _mm256_loadu_si256((void*)(&cur_table(j,i-1)));
    vecTabLeft[botlane] = _mm256_loadu_si256((void*)(&cur_table(j+1,i-1)));

    vecTabRight[toplane] = _mm256_loadu_si256((void*)(&cur_table(j-1,i+1)));
    vecTabRight[midlane] = _mm256_loadu_si256((void*)(&cur_table(j,i+1)));
    vecTabRight[botlane] = _mm256_loadu_si256((void*)(&cur_table(j+1,i+1)));

    vecTabMid[toplane] = _mm256_loadu_si256((void*)(&cur_table(j-1,i)));
    vecTabMid[midlane] = _mm256_loadu_si256((void*)(&cur_table(j,i)));
    vecTabMid[botlane] = _mm256_loadu_si256((void*)(&cur_table(j+1,i)));
    
    for ( j = y; j < y + height; j++){
      // printf("       %d / %d\n",j-y,height);
      // printf("\n\n\n=============New Line===========\n");
      // printf(" %d - %d / %d\n",i,i+height,y+height);
      // printVecLanes(vecTabMid,toplane,midlane,botlane,"MIDDLE\n");

      MtotVec = _mm256_add_epi8(\
                            _mm256_add_epi8(\
                              vecTabMid[toplane],\
                              vecTabMid[midlane]),\
                            vecTabMid[botlane]);
      LtotVec = _mm256_add_epi8(\
                            _mm256_add_epi8(\
                              vecTabLeft[toplane],\
                              vecTabLeft[midlane]),\
                            vecTabLeft[botlane]);
      RtotVec = _mm256_add_epi8(\
                            _mm256_add_epi8(\
                              vecTabRight[toplane],\
                              vecTabRight[midlane]),\
                            vecTabRight[botlane]);

      nVec =_mm256_add_epi8(MtotVec,\
              _mm256_add_epi8(LtotVec,RtotVec));
      
      // prntAVXi(nVec,"NVc");

      neq3 = _mm256_and_si256(_mm256_cmpeq_epi8(nVec,_mm256_set1_epi8(3)),mask1);
      meP3 = _mm256_add_epi8(vecTabMid[midlane],_mm256_set1_epi8(3));
      neqMeP3 = _mm256_and_si256(_mm256_cmpeq_epi8(nVec,meP3),mask1);

      nVec =  _mm256_or_si256(neqMeP3,neq3);
      change = _mm256_xor_si256(nVec,vecTabMid[midlane]);
      nVec = _mm256_and_si256(nVec,mask1);

      // printf("\nWho must live\n");
      // prntAVXi(nVec,"Nvc");

      _mm256_storeu_si256((void*)(&next_table(j,i)),nVec);
      
      bool vecChange = ! _mm256_testz_si256(_mm256_or_si256(change,_mm256_setzero_si256()), _mm256_set1_epi8(1));
      tileChange |= vecChange;

      //  printf("\n----rollout----\n");
      // printVecLanes(vecTabMid,toplane,midlane,botlane,"MIDDLE\n");
      // Rolling the roles
      vecTabRight[toplane]=_mm256_loadu_si256((void*)(&cur_table(j+2,i+1)));
      vecTabLeft[toplane]=_mm256_loadu_si256((void*)(&cur_table(j+2,i-1)));
      vecTabMid[toplane]=_mm256_loadu_si256((void*)(&cur_table(j+2,i)));
      cnt++;
      // printf("\n---rollout done\n");
      // printVecLanes(vecTabMid,toplane,midlane,botlane,"MIDDLE\n");
      // if(!_mm256_testz_si256(_mm256_xor_si256(nVec,_mm256_setzero_si256()), _mm256_set1_epi32(1)))
      //   exit(0);
    }
  }
  // if(tileChange)
  //   exit(0);

  return tileChange;
}

//must be called on tiles of width's sizes multiple of 32
static int do_tile_reg_vec (int x, int y, int width, int height)
{
  unsigned  tileChange = 0;
 
  __m256i vecTabLeft[3];
  __m256i vecTabMid[3];
  __m256i vecTabRight[3];
  __m256i MtotVec;
  __m256i LtotVec;
  __m256i RtotVec;

  __m256i neq3;
  __m256i meP3;
  __m256i neqMeP3;

  __m256i change;
  __m256i nVec;

  unsigned cnt = 0;
  
  #define toplane  ((cnt)%3)
  #define midlane  ((cnt+1)%3)
  #define botlane ((cnt+2)%3)
  
  unsigned i = x;
  

  for ( i = x; i < x + width; i+=VEC_SIZE){
    unsigned j = y;
    cnt = 0; //counts lines

    vecTabLeft[toplane] = _mm256_loadu_si256((void*)(&cur_table(j-1,i-1)));
    vecTabLeft[midlane] = _mm256_loadu_si256((void*)(&cur_table(j,i-1)));
    vecTabLeft[botlane] = _mm256_loadu_si256((void*)(&cur_table(j+1,i-1)));

    vecTabRight[toplane] = _mm256_loadu_si256((void*)(&cur_table(j-1,i+1)));
    vecTabRight[midlane] = _mm256_loadu_si256((void*)(&cur_table(j,i+1)));
    vecTabRight[botlane] = _mm256_loadu_si256((void*)(&cur_table(j+1,i+1)));

    vecTabMid[toplane] = _mm256_loadu_si256((void*)(&cur_table(j-1,i)));
    vecTabMid[midlane] = _mm256_loadu_si256((void*)(&cur_table(j,i)));
    vecTabMid[botlane] = _mm256_loadu_si256((void*)(&cur_table(j+1,i)));
    
    for ( j = y; j < y + height; j++){
      // printf("       %d / %d\n",j-y,height);
      // printf("\n\n\n=============New Line===========\n");
      // printf(" %d - %d / %d\n",i,i+height,y+height);
      // printVecLanes(vecTabMid,toplane,midlane,botlane,"MIDDLE\n");
      // printVecLanes(vecTabLeft,toplane,midlane,botlane,"LEFT\n");
      // printVecLanes(vecTabRight,toplane,midlane,botlane,"RIGHT\n");

      MtotVec = _mm256_add_epi8(\
                            _mm256_add_epi8(\
                              vecTabMid[toplane],\
                              vecTabMid[midlane]),\
                            vecTabMid[botlane]);
      LtotVec = _mm256_add_epi8(\
                            _mm256_add_epi8(\
                              vecTabLeft[toplane],\
                              vecTabLeft[midlane]),\
                            vecTabLeft[botlane]);
      RtotVec = _mm256_add_epi8(\
                            _mm256_add_epi8(\
                              vecTabRight[toplane],\
                              vecTabRight[midlane]),\
                            vecTabRight[botlane]);

      nVec =_mm256_add_epi8(MtotVec,\
              _mm256_add_epi8(LtotVec,RtotVec));
      
      // prntAVXi(nVec,"NVc");

      neq3 = _mm256_and_si256(_mm256_cmpeq_epi8(nVec,_mm256_set1_epi8(3)),mask1);
      meP3 = _mm256_add_epi8(vecTabMid[midlane],_mm256_set1_epi8(3));
      neqMeP3 = _mm256_and_si256(_mm256_cmpeq_epi8(nVec,meP3),mask1);

      nVec =  _mm256_or_si256(neqMeP3,neq3);
      change = _mm256_xor_si256(nVec,vecTabMid[midlane]);
      nVec = _mm256_and_si256(nVec,mask1);

      // printf("\nWho must live\n");
      // prntAVXi(nVec,"Nvc");

      _mm256_storeu_si256((void*)(&next_table(j,i)),nVec);
      
      bool vecChange = ! _mm256_testz_si256(_mm256_or_si256(change,_mm256_setzero_si256()), _mm256_set1_epi8(1));
      tileChange |= vecChange;

      //  printf("\n----rollout----\n");
      // printVecLanes(vecTabMid,toplane,midlane,botlane,"MIDDLE\n");
      // printVecLanes(vecTabLeft,toplane,midlane,botlane,"LEFT\n");
      // printVecLanes(vecTabRight,toplane,midlane,botlane,"RIGHT\n");
      // Rolling the roles
      vecTabRight[toplane]=_mm256_loadu_si256((void*)(&cur_table(j+2,i+1)));
      vecTabLeft[toplane]=_mm256_loadu_si256((void*)(&cur_table(j+2,i-1)));
      vecTabMid[toplane]=_mm256_loadu_si256((void*)(&cur_table(j+2,i)));
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

  if(ENABLE_BITCELL)
    r = do_tile_reg_vec_bits (x, y, width, height);
  else
    r = do_tile_reg_vec (x, y, width, height);

  monitoring_end_tile (x, y, width, height, who);

  return r;
}

static bool hasAnyTileChanged(void){
  for(int i =0 ; i < 256; i ++){
    if(threadChange[i]){
      return true;
    }
  }
  return false;
}


//////////////////////// BitMap2 lazy vectorial Version ;
//OMP_PLACES=cores OMP_NUM_THREADS=1 ./run -k life -s 2048 -ts 64 -v lazybtmpvec -m
//OMP_PLACES=cores OMP_NUM_THREADS=1 ./run -k life -a random -s 2048 -ts 32 -v lazybtmpvec -m
//OMP_PLACES=cores OMP_NUM_THREADS=4 ./run -k life -a otca_off -s 2176 -ts 64 -v lazybtmpvec -m

#define xgrain 2 

unsigned life_compute_lazybtmpvec (unsigned nb_iter){ 
    
  unsigned x;
  unsigned y;
  unsigned who;
  unsigned itres = 0;
  unsigned res;

  for(unsigned it=1; it<=nb_iter;it++){

    #pragma omp parallel for schedule(dynamic) private(res,x,y,who)
    for(int j = 0; j< NB_TILES_Y * xgrain ;j++){ 
      for(int i = 0; i< (NB_TILES_X/xgrain)  ;i++){
        int column = j/NB_TILES_Y;
        int jj = j%NB_TILES_Y;
        int ii = column * (NB_TILES_X/xgrain) + i;

        if(hasNeighbourChanged(ii,jj)){
          x=ii * TILE_W;
          y=jj * TILE_H;
          who = omp_get_thread_num();
          
          res = do_tile_vec(x, y , TILE_W, TILE_H, who);
          
          threadChange[omp_get_thread_num()] |= res;
          next_map(ii,jj)=res;      
        }
      } 
    }
    deleteCurrentBtmp();
    swap_tables ();
    if (!hasAnyTileChanged()) { // we stop when all cells are stable
      itres = it;
      printf("there's no future tasks\n");
      //break;
    }
  }
return itres;
}

//////////////////////////////// Tiled omp version
// OMP_PLACES=cores OMP_NUM_THREADS=1 ./run -k life -a random -s 1024 -ts 128 -i 100 -n -v ompvec
unsigned life_compute_ompvec (unsigned nb_iter)
{
  unsigned res = 0;
  for (unsigned it = 1; it <= nb_iter; it++) {
    unsigned change = 0;
    #pragma omp parallel for schedule (dynamic) shared (change)
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

//////////////////////////////// Tiled omp version
// OMP_PLACES=cores OMP_NUM_THREADS=1 ./run -k life -a random -s 1024 -ts 128 -i 100 -n -v omp
unsigned life_compute_omp (unsigned nb_iter)
{
  unsigned res = 0;

  for (unsigned it = 1; it <= nb_iter; it++) {
    unsigned change = 0;
    #pragma omp parallel for schedule (dynamic)
    for (int y = 0; y < DIM; y += TILE_H)
      for (int x = 0; x < DIM; x += TILE_W){
        change |= do_tile_nocheck (x, y, TILE_W, TILE_H, omp_get_thread_num());
      }
    swap_tables ();
    
    if (!change) { // we stop when all cells are stable
      res = it;
      break;
    }
  }
  return res;
}

//////////////////////////////// Sequential version
// ./run -k random -s 1024 -ts 128 -i 100 -n -v omp
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

unsigned life_compute_bits(unsigned nb_iter)
{
  int change = 0;
  unsigned it = 1;
  for (; it <= nb_iter; it++) {
    //printTable(_table);
    //clear_table(_alternate_table);
    monitoring_start_tile (0);

    for (int i = 0; i < DIM+1; i++)
      for (int j = 0; j < DIM; j++)
        change |= compute_byte (i, j);
        
    monitoring_end_tile (0, 0, DIM, DIM, 0);
    
    swap_tables ();
    if (!change)
      return it;
  }
  return 0 ;
}

static inline void swap_buffers(void)
{
  cl_mem tmp2  = cur_buffer;
  cur_buffer = next_buffer;
  next_buffer = tmp2;
}

///////////////////////////// OpenCL big variant (ocl_big)
// ./run -k life -o
unsigned life_invoke_ocl (unsigned nb_iter)
{
  size_t global[2] = {DIM-2,DIM-2};
  size_t local[2]  = {GPU_TILE_W,GPU_TILE_H};
  cl_int err;

  bool gpuChange = False;

  cl_mem changeBuffer = clCreateBuffer (context, CL_MEM_READ_WRITE,
                            sizeof (unsigned), NULL, NULL);

                
  monitoring_start_tile (easypap_gpu_lane (TASK_TYPE_COMPUTE));

  for (unsigned it = 1; it <= nb_iter; it++) {
    
    gpuChange = False;
    err = 0;

    // Modifies the buffer
    err =clEnqueueWriteBuffer (queue, changeBuffer, CL_TRUE, 0,
             sizeof (unsigned int), &gpuChange, 0, NULL,NULL); 
    check (err, "Failed to write the buffer");

    // Sets Kernel arguments
    err |= clSetKernelArg (compute_kernel, 0,  sizeof (cl_mem), &cur_buffer);
    err |= clSetKernelArg (compute_kernel, 1,  sizeof (cl_mem), &next_buffer);
    err |= clSetKernelArg (compute_kernel, 2,  sizeof (cl_mem), &changeBuffer);
    check (err, "Failed to set kernel arguments");

    // Launches GPU kernel
    err = clEnqueueNDRangeKernel (queue, compute_kernel, 2, NULL, global, NULL,
                                  0, NULL, NULL);
    check (err, "Failed to execute kernel");

    // Retrieves the Buffer
    err = clEnqueueReadBuffer(queue, changeBuffer, CL_TRUE, 0, sizeof(unsigned),
          &gpuChange, 0, NULL, NULL);
    check (err, "Failed to Read the buffer");

    if(!gpuChange)
      return it;

    swap_buffers();
    swap_tables();
  }

  clFinish (queue);

  monitoring_end_tile (0, 0, DIM, DIM, easypap_gpu_lane (TASK_TYPE_COMPUTE));

  return 0;
}

unsigned life_invoke_ocl2 (unsigned nb_iter)
{
  size_t global[2] = {DIM*3,DIM*3};
  size_t local[2]  = {3,3};
  cl_int err;

  bool gpuChange = False;

  cl_mem changeBuffer = clCreateBuffer (context, CL_MEM_READ_WRITE,
                            sizeof (unsigned), NULL, NULL);
  
  monitoring_start_tile (easypap_gpu_lane (TASK_TYPE_COMPUTE));

  for (unsigned it = 1; it <= nb_iter; it++) {
    
    
    err = 0;

    // Modifies the buffer
    if(!(it%100)){
    gpuChange = False;
    err |=clEnqueueWriteBuffer (queue, changeBuffer, CL_TRUE, 0,
             sizeof (unsigned int), &gpuChange, 0, NULL,NULL); 
    check (err, "Failed to write the buffer");
    }

    // Sets Kernel arguments
    err |= clSetKernelArg (compute_kernel, 0,  sizeof (cl_mem), &cur_buffer);
    err |= clSetKernelArg (compute_kernel, 1,  sizeof (cl_mem), &next_buffer);
    err |= clSetKernelArg (compute_kernel, 2,  sizeof (cl_mem), &changeBuffer);
    check (err, "Failed to set kernel arguments");

    // Launches GPU kernel
    err |= clEnqueueNDRangeKernel (queue, compute_kernel, 2, NULL, global, local,
                                  0, NULL, NULL);
    check (err, "Failed to execute kernel");

    if(!(it%100)){
      //Retrieves the Buffer
      err |= clEnqueueReadBuffer(queue, changeBuffer, CL_TRUE, 0, sizeof(unsigned),
            &gpuChange, 0, NULL, NULL);
      check (err, "Failed to Read the buffer");

      if(!gpuChange)
        return it;
    }
    swap_buffers();
    swap_tables();
  }

  clFinish (queue);

  monitoring_end_tile (0, 0, DIM, DIM, easypap_gpu_lane (TASK_TYPE_COMPUTE));

  return 0;
}
///////////////////////////// Initial configs

void life_draw_guns (void);

static inline void set_cell (int y, int x)
{
  if(ENABLE_BITCELL)
    setcurBitCell(x,y,1);
  else
    cur_table (y, x) = 1;
  if (opencl_used)
    cur_img (y, x) = 1;
}


static inline int get_cell (int y, int x)
{
  if(ENABLE_BITCELL)
    return getBitCell(x,y);
  return cur_table (y,x);
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