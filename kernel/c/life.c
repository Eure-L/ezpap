#include "easypap.h"
#include "rle_lexer.h"
#include "arch_flags.h"

#include <avxintrin.h>
#include <omp.h>
#include <stdbool.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

/////// Changer le type en unsigned pour la version en bits
typedef char cell_t;
///////

static unsigned color = 0xFFFF00FF; // Living cells have the yellow color

static cell_t *restrict _table = NULL, *restrict _alternate_table = NULL;
char * bitMapTls; // Two Bit maps represented in a vector
                  // Each bits representing a tile to compute
bool switcher = 1;
__m256i mask1;
__m256i zero;
char threadChange[256];

#define AVXBITS 256
#define VEC_SIZE (AVXBITS/(sizeof(cell_t)*8))

#define NB_TILES_TOT (NB_TILES_X*NB_TILES_Y)
#define NB_FAKE_X (NB_TILES_X + 2)
#define NB_FAKE_Y (NB_TILES_Y + 2)
#define NB_FAKE_TILES ((NB_FAKE_X)*(NB_FAKE_Y))
#define curTable switcher
#define nextTable !switcher

unsigned bits; // nb of bits in cell_t type
unsigned ENABLE_BITCELL = 0;
unsigned SIZEX ;
unsigned SIZEY ;


#define _table_SIZE (SIZEX*SIZEY*sizeof(cell_t))
#define DIMTOT (SIZEX*SIZEY)

cl_mem changeBuffer;
cl_mem curbitMapBuffer;
cl_mem nextbitMapBuffer;
cl_event transfert_event;

static unsigned cpu_y_part;
static unsigned gpu_y_part;
// Threashold
#define THRESHOLD 10

static inline cell_t *table_cell (cell_t *restrict i, int y, int x)
{
  //added empty border for optimized structure for AVX usage
  return i + (y+1) * (DIM+VEC_SIZE*2) + (x + VEC_SIZE);
}

static inline cell_t *table_cell_row (cell_t *restrict i, int y, int x)
{ 
  return i + (y+1) * (SIZEX) + (((x+bits)/bits)+VEC_SIZE);
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
#define next_tableAddr(y, x) (table_cell (_alternate_table, (y), (x)))

#define cur_map(x, y) (*table_map ((bitMapTls+(curTable*NB_FAKE_TILES)), (x), (y)))
#define cur_fmap(x, y) (*fake_map ((bitMapTls+(curTable*NB_FAKE_TILES)), (x), (y)))

#define cur_mapAddr(x, y) (table_map ((bitMapTls+(curTable*NB_FAKE_TILES)), (x), (y)))
#define cur_fmapAddr(x, y) (fake_map ((bitMapTls+(curTable*NB_FAKE_TILES)), (x), (y)))

#define cur_table_row(y, x) ((*table_cell_row (_table, (y), (x))))
#define next_table_row(y, x) (*table_cell_row (_alternate_table, (y), (x)))

#define next_map(x, y) (*table_map ((bitMapTls+(nextTable*NB_FAKE_TILES)), (x), (y)))
#define next_fmap(x, y) (*fake_map ((bitMapTls+(nextTable*NB_FAKE_TILES)), (x), (y)))
#define next_mapAddr(x, y) (table_map ((bitMapTls+(nextTable*NB_FAKE_TILES)), (x), (y)))
#define next_fmapAddr(x, y) (fake_map ((bitMapTls+(nextTable*NB_FAKE_TILES)), (x), (y)))


static inline void setcurBitCellRow(int i, int j, unsigned val){
  cur_table_row(j,i) = 
  (cur_table_row(j,i) & ~(0x01<<((i)%bits))) | (val<<((i)%bits));
}

static inline void OCLsetcurBitCellRow(int i, int j, unsigned val){
  cur_img(j,i) = 
  (cur_img(j,i) & ~(0x01<<((i)%bits))) | (val<<((i)%bits));
}

static inline void setnextBitCellRow(int i, int j, unsigned val){
  next_table_row(j,i) = 
  ((next_table_row(j,i)& ~(0x01<<((i%bits)))) | (val<<((i)%bits)));
}

static inline unsigned getBitCellRow(int i, int j){
  return (cur_table_row(j,i)>>((i)%bits)&(0x01));
}

// This function is called whenever the graphical window needs to be refreshed
void life_refresh_img (void)
{
  for (int i = 0; i < DIM; i++)
    for (int j = 0; j < DIM; j++)
      cur_img (i, j) = cur_table (i, j) * color;
}

void life_refresh_img_bitbrd (void)
{
  for (int i = 0; i < DIM; i++){
    for (int j = 0; j < DIM; j++){
        cur_img (i, j) = (getBitCellRow (j, i) )* color; 
    }
  }
}

void life_refresh_img_ocl_bits (void){
  cl_int err;
  err = clEnqueueReadBuffer (queue, cur_buffer, CL_TRUE, 0, _table_SIZE, _table, 0, NULL, NULL);
  check (err, "Failed to read buffer from GPU");
  life_refresh_img_bitbrd();
}

void life_refresh_img_ocl(void){
  cl_int err;
  err = clEnqueueReadBuffer (queue, cur_buffer, CL_TRUE,0,
                             _table_SIZE, _table, 
                             0, NULL, NULL);
  check (err, "Failed to read buffer from GPU");
  life_refresh_img();
}

void life_refresh_img_ocl_hybrid (void){

  cl_int err;
  unsigned offset =  (cpu_y_part+1) * SIZEX;
  err = clEnqueueReadBuffer (queue, cur_buffer, CL_TRUE,offset,
                             gpu_y_part*SIZEX, _table+offset, 
                             0, NULL, NULL);
  check (err, "Failed to read buffer from GPU");
  life_refresh_img();
}


char * initBtmptls(void){
  
  char * map = (char *) malloc ((2 * (NB_FAKE_TILES) * sizeof(char)));
  if(map == NULL){
    printf("map pointer NULL\n");
    exit(EXIT_FAILURE);
  }       

  for (int i=0;i<NB_FAKE_TILES;i++){
    *(map+i)=0;
  }
  for (int i=0;i<NB_FAKE_TILES;i++){
    *(map+NB_FAKE_TILES+i)=1;
  }
  
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
  for(int j = 0;j<NB_FAKE_Y; j++){
    for(int i = 0; i<NB_FAKE_X; i++){
      cur_fmap(i,j)=0;
    }
  }
}

/**
 * @brief Sets the GPU border side to 1 for cpulazy evaluation
 */
void setGPUborderBtmp(){
  int j=cpu_y_part/TILE_H;
  for(int i = 0; i<NB_FAKE_X; i++){
    next_fmap(i,j+1)=1;
  }
}

void clear_table(cell_t * table){
  for(int i =0; i<DIMTOT;i++)
    *(table+i)=0;
}

/**
 * @brief Used to check if any adjacent tile has changed
 * 
 * @param i 
 * @param j 
 * @return true 
 * @return false 
 */
static inline bool hasNeighbourChanged(unsigned i,unsigned j){
  //printf("isok\n");
  // #define cur_map(y, x) (*table_map ((bitMapTls+(curTable*NB_FAKE_TILES)), (y), (x)))
  
  return cur_map(i,j)|cur_map(i-1,j)|cur_map(i+1,j)|cur_map(i,j-1)|cur_map(i,j+1)
  |cur_map(i-1,j-1)|cur_map(i+1,j+1)|cur_map(i+1,j-1)|cur_map(i-1,j+1);    
}

void life_init (void)
{
  SIZEX =(DIM+(VEC_SIZE*2));
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
  ENABLE_BITCELL = 0;
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
  life_init();
  changeBuffer = clCreateBuffer (context, CL_MEM_READ_WRITE,
                            sizeof (unsigned), NULL, NULL);

}

void life_init_bitbrd(void)
{
  ENABLE_BITCELL = 2;
  bits = sizeof(cell_t)*8;
  SIZEX =(DIM+(2*(VEC_SIZE)))/bits;
  SIZEY =(DIM)+2;
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
  
  clear_table(_table);
}

void life_init_ocl_bits(void){

  bits = sizeof(cell_t)*8;
  SIZEX =(DIM/bits);
  SIZEY =(DIM);
  if (_table == NULL) {
    const unsigned size = _table_SIZE*2;
    PRINT_DEBUG ('u', "Memory footprint = 2 x %d bytes\n", size);

    _table = mmap (NULL, size, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0 );

    _alternate_table = mmap (NULL, size, PROT_READ | PROT_WRITE,
                             MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  }
  if(bitMapTls==NULL)
    bitMapTls = initBtmptls(); 
  
  changeBuffer = clCreateBuffer (context, CL_MEM_READ_WRITE,
                            sizeof (unsigned), NULL, NULL);
  ENABLE_BITCELL = 2;

}

/**
 * @brief Called upon each end of iteration to swap tables
 * 
 */
static inline void swap_tables (void)
{
  cell_t *tmp = _table;
  switcher = !switcher;
  _table           = _alternate_table;
  _alternate_table = tmp;
}

/**
 * @brief Prints a vector alongside its given name
 * 
 * @param vec 
 * @param name 
 */
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

void printVecLanes(__m256i * vecLst, int topL, int midL, int botL,char * str){
  printf("%s",str);
  prntAVXi(vecLst[topL],"top");
  prntAVXi(vecLst[midL],"mid");
  prntAVXi(vecLst[botL],"bot");
}

///////////////////////////// Sequential version (seq)

static int compute_new_state (int y, int x)
{
  unsigned n      = 0;
  unsigned me     = cur_table (y, x) != 0;
  unsigned change = 0;

  if (x > 0 && x < DIM - 1 && y > 0 && y < DIM - 1) {

    n += cur_table (y,x);
    n += cur_table (y-1,x);
    n += cur_table (y,x-1);
    n += cur_table (y-1,x-1);
    n += cur_table (y+1,x);
    n += cur_table (y,x+1);
    n += cur_table (y+1,x+1);
    n += cur_table (y+1,x-1);
    n += cur_table (y-1,x+1);

    n = (n == 3 + me) | (n == 3);
    if (n != me)
      change |= 1;

    next_table (y, x) = n;
  }
  return change;
}

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


/////////////////////////////////////////// vectorial version

//must be called on tiles of width's sizes multiple of 32
static int do_tile_reg_vec (int x, int y, int width, int height)
{
  unsigned  tileChange = 0;
  
  // adjacents vectors
  __m256i vecTabLeft[3];
  __m256i vecTabMid[3];
  __m256i vecTabRight[3];
  __m256i MtotVec;
  __m256i LtotVec;
  __m256i RtotVec;

  // for logical operation
  __m256i neq3;
  __m256i meP3;
  __m256i neqMeP3;

  __m256i change;
  __m256i nVec;

  __m256i tmpLines[TILE_H][width/VEC_SIZE];

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
      
      neq3 = _mm256_and_si256(_mm256_cmpeq_epi8(nVec,_mm256_set1_epi8(3)),mask1);
      meP3 = _mm256_add_epi8(vecTabMid[midlane],_mm256_set1_epi8(3));
      neqMeP3 = _mm256_and_si256(_mm256_cmpeq_epi8(nVec,meP3),mask1);

      nVec =  _mm256_or_si256(neqMeP3,neq3);
      change = _mm256_xor_si256(nVec,vecTabMid[midlane]);
      nVec = _mm256_and_si256(nVec,mask1);;

      // printf("j %d\n",j);
      // printf("i %d\n",(i-x));
      // printf("VECSIZE %d\n",VEC_SIZE);
      // printf("indice %d\n",(i-x)/VEC_SIZE);
      tmpLines[j-y][(i-x)/VEC_SIZE] = nVec;
      //_mm256_storeu_si256((void*)(next_tableAddr(j,i)),nVec);
      
      bool vecChange = ! _mm256_testz_si256(_mm256_or_si256(change,_mm256_setzero_si256()), _mm256_set1_epi8(1));
      tileChange |= vecChange;

      // Rolling the roles
      vecTabRight[toplane]=_mm256_loadu_si256((void*)(&cur_table(j+2,i+1)));
      vecTabLeft[toplane]=_mm256_loadu_si256((void*)(&cur_table(j+2,i-1)));
      vecTabMid[toplane]=_mm256_loadu_si256((void*)(&cur_table(j+2,i)));
      cnt++;
    }
  }
  for(int j=0; j<TILE_H; j++){
    for(int i=0; i<width/VEC_SIZE; i++){
      _mm256_storeu_si256((void*)(next_tableAddr( y+j , x+i*VEC_SIZE )),tmpLines[j][i]);
    }
  }
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

static int do_tile_reg_bitbrd(int x, int y, int width, int height)
{ 
  cell_t vecTabLeft[3];
  cell_t vecTabMid[3];
  cell_t vecTabRight[3];
  cell_t s2;
  cell_t s3;
  cell_t a,b,c,d,e,f,g,h;
  cell_t X;
  cell_t XAB;
  cell_t XCD;
  cell_t XEF;
  cell_t XGH;

  cell_t change = false;
  cell_t res;

  unsigned cnt = 0;
  
  #define toplane  ((cnt)%3)
  #define midlane  ((cnt+1)%3)
  #define botlane ((cnt+2)%3)
  
  unsigned i = x;
  

  for (i = x; i < x + width; i+= bits){

    unsigned j = y;
    cnt = 0; //counts lines

    vecTabMid[toplane] = cur_table_row(j-1,i);
    vecTabMid[midlane] = cur_table_row(j,i);
    vecTabMid[botlane] = cur_table_row(j+1,i);
    
    // to get the vectors on the left side, middle vectors are shifted to the
    // right and we add the last bit that was out of scope
    vecTabLeft[toplane] = (vecTabMid[toplane]<<1)|(getBitCellRow(i-1,j-1));
    vecTabLeft[midlane] = (vecTabMid[midlane]<<1)|(getBitCellRow(i-1,j));
    vecTabLeft[botlane] = (vecTabMid[botlane]<<1)|(getBitCellRow(i-1,j+1));

    // to get the vectors on the right side, middle vectors are shifted to the
    // left and we add the last bit that was out of scope
    vecTabRight[toplane] = (vecTabMid[toplane]>>1)|((getBitCellRow(i+bits,j-1)<<(bits-1)));
    vecTabRight[midlane] = (vecTabMid[midlane]>>1)|((getBitCellRow(i+bits,j)<<(bits-1)));
    vecTabRight[botlane] = (vecTabMid[botlane]>>1)|((getBitCellRow(i+bits,j+1)<<(bits-1)));
    
    for (int j = y; j < y + height; j++){

      XAB = vecTabLeft[toplane] & vecTabMid[toplane];
      a   = vecTabLeft[toplane] ^ vecTabMid[toplane];
      XCD = vecTabRight[toplane] & vecTabLeft[midlane];
      c   = vecTabRight[toplane] ^ vecTabLeft[midlane];
      XEF = (vecTabRight[midlane] & vecTabLeft[botlane]);
      e   = (vecTabRight[midlane] ^ vecTabLeft[botlane]);
      XGH = (vecTabMid[botlane] &  vecTabRight[botlane]);
      g   =  (vecTabMid[botlane] ^  vecTabRight[botlane]);

      d =  ( a & c);
      a =  (a ^ c);
      c =  (XAB & XCD);
      b =  (XAB ^  (XCD ^ d));

      h =  ( e & g);
      e =  (e ^ g);
      g =  (XEF & XGH);
      f =   (XEF ^  (XGH ^ h));

      d =  (a & e);
      a =  (a ^ e);
      h =  (b & f);
      b =  (b ^ f);
      h =  (h |  (b & d));
      b =  (b ^ d);
      c =  (c ^  (g ^ h));

      X =  ( ~c  & b);
      s2=  (X & ~a);
      s3 =  (X & a);

      res =  (s3 |  (vecTabMid[midlane] & s2));

      next_table_row(j,i) = res;

      change |= !(vecTabMid[midlane] == res);

      vecTabMid[toplane]   = cur_table_row(j+2,i);
      vecTabRight[toplane] = (vecTabMid[toplane]>>1)|((getBitCellRow(i+bits,j+2)<<(bits-1)));
      vecTabLeft[toplane]  = (vecTabMid[toplane]<<1)|(getBitCellRow(i-1,j+2));
      cnt++;
    }
  }
  return change;
}

static int do_tile_bitbrdvec (int x, int y, int width, int height, int who)
{
  int r;

  monitoring_start_tile (who);

  r = do_tile_reg_bitbrd(x, y, width, height);

  monitoring_end_tile (x, y, width, height, who);

  return r;
}


// returns true if any thread has witnessed any change
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


unsigned life_compute_lazybtmpvec (unsigned nb_iter){ 
    
  unsigned x;
  unsigned y;
  unsigned who;
  unsigned itres = 0;
  unsigned res;

  for(unsigned it=1; it<=nb_iter;it++){

    #pragma omp parallel for schedule(dynamic) private(res,x,y,who)
    for(int j = 0; j< NB_TILES_Y ;j++){ 
      for(int i = 0; i< NB_FAKE_X  ;i++){

        if(hasNeighbourChanged(i,j)){
          x=i * TILE_W;
          y=j * TILE_H;
          who = omp_get_thread_num();
          res = do_tile_vec(x, y , TILE_W, TILE_H, who);
          
          threadChange[omp_get_thread_num()] |= res;
          *next_mapAddr(i,j)=res;      
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

//////////////////////// Bit board Version ;
//OMP_PLACES=cores OMP_NUM_THREADS=4 ./run -k life -s 2048 -ts 64 -v bitbrd -m
unsigned life_compute_bitbrd(unsigned nb_iter)
{
  unsigned x;
  unsigned y;
  unsigned who;
  unsigned res;
  unsigned it = 1;
  for (; it <= nb_iter; it++) {
    #pragma omp parallel for schedule(dynamic) private(res,x,y,who)
    for (int i = 0; i < NB_TILES_Y; i++){
      for (int j = 0; j < NB_TILES_X; j++){
        if(hasNeighbourChanged(j,i)){
          who =omp_get_thread_num();
          x = j * TILE_W;
          y = i * TILE_H;
          who = omp_get_thread_num();
          res = do_tile_bitbrdvec(x,y,TILE_W,TILE_H,who);
          next_map(j,i)=res;
        }
      }
    }
    deleteCurrentBtmp();
    swap_tables ();
  }
  return 0;
}


static inline void swap_buffers(void)
{ 
  cl_mem tmp1 = curbitMapBuffer;
  curbitMapBuffer = nextbitMapBuffer;
  nextbitMapBuffer = tmp1;

  cl_mem tmp2  = cur_buffer;
  cur_buffer = next_buffer;
  next_buffer = tmp2;
}

///////////////////////////// OpenCL  variant (ocl)
// ./run -k life -o
unsigned life_invoke_ocl (unsigned nb_iter)
{
  size_t global[2] = {(DIM),(DIM)};
  //size_t local[2]  = {3,3};
  cl_int err;

  bool gpuChange = False;
                
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
    //check (err, "Failed to Read the buffer");

    if(!gpuChange)
      return it;

    swap_buffers();
    swap_tables();
  }

  clFinish (queue);

  monitoring_end_tile (0, 0, DIM, DIM, easypap_gpu_lane (TASK_TYPE_COMPUTE));

  return 0;
}

unsigned life_invoke_ocl_bits(unsigned nb_iter){

  size_t global[2] = {(DIM/bits),(DIM)};

  cl_int err; 
  err = clEnqueueWriteBuffer(queue, cur_buffer, CL_TRUE, 0, _table_SIZE, _table, 0, NULL, NULL);
  check (err, "Failed to write buffer to GPU");

  bool gpuChange = False;
                
  monitoring_start_tile (easypap_gpu_lane (TASK_TYPE_COMPUTE));

  for (unsigned it = 1; it <= nb_iter; it++) {
    gpuChange = False;
    err = 0;

    // Modifies the buffer
    err =clEnqueueWriteBuffer (queue, cur_buffer, CL_TRUE, 0,
            _table_SIZE, _table, 0, NULL,NULL);

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

    err = clEnqueueReadBuffer (queue, cur_buffer, CL_TRUE, 0, _table_SIZE, _table, 0, NULL, NULL);
    check (err, "Failed to read buffer from GPU");
    //check (err, "Failed to Read the buffer");

    if(false)
      return it;

    swap_buffers();
    swap_tables();
  }

  clFinish (queue);

  monitoring_end_tile (0, 0, DIM, DIM, easypap_gpu_lane (TASK_TYPE_COMPUTE));

  return 0;

}



static long gpu_duration = 0, cpu_duration = 0;


static int much_greater_than (long t1, long t2)
{
  return (t1 > t2) && ((t1 - t2) * 100 / t1 > THRESHOLD);
}

static cl_int nqWrtBuf(unsigned offset, unsigned size, cl_mem * buffer, cell_t * ptr){
  return clEnqueueWriteBuffer (queue, *buffer, CL_TRUE, 
                                  offset,                 //offset write buffer
                                  size,                   //size 
                                  ptr,                    //pointer
                                  0,
                                  NULL, &transfert_event);
}

static cl_int nqRdBuf(unsigned offset, unsigned size, cl_mem * buffer, cell_t * ptr){
  return clEnqueueReadBuffer (queue, *buffer, CL_TRUE, 
                                  offset,                 //offset 
                                  size,                   //size 
                                  ptr,                    //pointer
                                  0,
                                  NULL, &transfert_event);
}

static void cpu_write_all(){
  cl_int err;
  unsigned rows       = cpu_y_part;
  unsigned offset     = SIZEX * sizeof (cell_t);
  unsigned data_size  = SIZEX * rows * sizeof (cell_t);
  cell_t * ptr        = _alternate_table + offset;

  err = nqWrtBuf(offset,  data_size, &next_buffer, ptr);
  ocl_monitor (transfert_event, 0, cpu_y_part-rows, DIM, rows, TASK_TYPE_WRITE);
  check (err, "Failed to send (part of) CPU contribution to the buffer");
}

static void do_statusQuo(){
  cl_int err;
  unsigned rows       = 1;
  unsigned offset     = SIZEX * (cpu_y_part+1 - rows);
  unsigned data_size  = SIZEX  * rows * sizeof (cell_t);
  cell_t * ptr        = _alternate_table + offset;

  //  CPU_PART
  if((!do_display)){
    err = nqWrtBuf(offset,  data_size, &next_buffer, ptr);
    ocl_monitor (transfert_event, 0, cpu_y_part+1-rows, DIM, rows, TASK_TYPE_WRITE);
    check (err, "Failed to send (part of) CPU contribution to the buffer");
  }else{
    cpu_write_all();
  }

  //GPU part
  offset    = SIZEX * (cpu_y_part+1) * sizeof (cell_t);
  ptr       = _alternate_table + offset;
  err       = nqRdBuf(offset,  data_size, &next_buffer, ptr);
  ocl_monitor (transfert_event, 0, cpu_y_part+1, DIM, rows, TASK_TYPE_READ);
  check (err, "Failed to send GPU bordering tiles contribution to the RAM");
}

// if the GPU steals load
// there is no need for GPU => CPU data transfer
static void gpu_steals_load(){
  cl_int err;
  unsigned rows       = TILE_H  + 1 ;
  unsigned offset     = SIZEX   * (cpu_y_part+1 - rows);
  unsigned data_size  = SIZEX   * rows * sizeof (cell_t);
  cell_t * ptr        = _alternate_table + offset ;
  
  if((!do_display)){
    err = nqWrtBuf(offset,  data_size, &next_buffer, ptr);
    ocl_monitor (transfert_event, 0, cpu_y_part+1-rows, DIM, rows, TASK_TYPE_WRITE);
    check (err, "Failed to send (part of) CPU contribution to the buffer");
  }else{
    cpu_write_all();
  }
  
  PRINT_DEBUG ('z', "CPU %.2f%%     /     GPU %.2f%% \n", (float)cpu_y_part/DIM*100,(float)gpu_y_part/DIM*100);
  PRINT_DEBUG ('z', "                         ++\n");
}

// if the CPU steals load
// there is no need for CPU => GPU data transfer
static void cpu_steals_load(){
  cl_int err;
  unsigned rows       = TILE_H  +1;
  unsigned offset     = SIZEX   * (cpu_y_part+1) * sizeof (cell_t);
  unsigned data_size  = SIZEX   * rows * sizeof (cell_t);
  cell_t * ptr        = _alternate_table + SIZEX * (cpu_y_part+1) * sizeof (cell_t);     
  
  err = nqRdBuf(offset,  data_size, &next_buffer, ptr);
  ocl_monitor (transfert_event, 0, cpu_y_part+1, DIM, rows, TASK_TYPE_READ);
  check (err, "Failed to send GPU bordering tiles contribution to the RAM");
  clFinish (queue);

  if(do_display){
    cpu_write_all();
  }
  
  PRINT_DEBUG ('z', "CPU %.2f%%     /     GPU %.2f%%  \n", (float)cpu_y_part/DIM*100,(float)gpu_y_part/DIM*100);
  PRINT_DEBUG ('z', "    ++\n");
}



void life_init_ocl_hybrid(void){
  
  bits  = sizeof(cell_t)*8;
  SIZEX =(DIM+(VEC_SIZE*2));
  SIZEY =(DIM+2);
  if (_table == NULL) {
    const unsigned size = _table_SIZE*2;
    PRINT_DEBUG ('u', "Memory footprint = 2 x %d bytes\n", size);

    _table = mmap (NULL, size, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0 );

    _alternate_table = mmap (NULL, size, PROT_READ | PROT_WRITE,
                             MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  }
  if(bitMapTls==NULL)
    bitMapTls = initBtmptls(); 
  
  changeBuffer = clCreateBuffer (context, CL_MEM_READ_WRITE,
                            sizeof (unsigned), NULL, NULL);
  zero = _mm256_setzero_si256(); 
  mask1 = _mm256_set1_epi8(1);
  ENABLE_BITCELL = 0;

  if (GPU_TILE_H != TILE_H)
    exit_with_error ("CPU and GPU Tiles should have the same height (%d != %d)",
                    GPU_TILE_H, TILE_H);

  cpu_y_part = ((NB_TILES_Y*4) / 10) * GPU_TILE_H; // Start with sixty-fourty
  gpu_y_part = DIM - cpu_y_part;

}



unsigned life_invoke_ocl_hybrid (unsigned nb_iter)
{
  unsigned gpu_accumulated_lines = 0;
  //GPU VARIABLES
  size_t global[2] = {DIM, gpu_y_part};
  size_t local[2]  = {GPU_TILE_W, GPU_TILE_H};
  cl_int err;
  cl_event kernel_event;
  bool gpuChange = False;

  //CPU VARIABLES
  unsigned x;
  unsigned y;
  unsigned who;
  unsigned res;
  long t1,t2;
  for (unsigned it = 1; it <= nb_iter; it++) {
    
    gpuChange = False;
    err = 0;

    // sets gpuChange status into GPU memory
    err =clEnqueueWriteBuffer (queue, changeBuffer, CL_TRUE, 0,
             sizeof (unsigned int), &gpuChange, 0, NULL,&kernel_event); 
    check (err, "Failed to write the Changebuffer");

    // Sets Kernel arguments
    err |= clSetKernelArg (compute_kernel, 0,  sizeof (cl_mem), &cur_buffer);
    err |= clSetKernelArg (compute_kernel, 1,  sizeof (cl_mem), &next_buffer);
    err |= clSetKernelArg (compute_kernel, 2,  sizeof (cl_mem), &changeBuffer);
    check (err, "Failed to set kernel arguments");
    err |= clSetKernelArg (compute_kernel, 3, sizeof (unsigned), &cpu_y_part);
    check (err, "Failed to set kernel arguments");
    // Launches GPU kernel
    err = clEnqueueNDRangeKernel (queue, compute_kernel, 2, NULL, global, local,
                                  0, NULL, &kernel_event);
    check (err, "Failed to execute kernel");
    clFlush (queue);

     /////// CPU part
    t1 = what_time_is_it ();
    #pragma omp parallel for schedule(dynamic) private(res,x,y,who)
    for(int j = 0; j< cpu_y_part/TILE_H  ;j++){ 
      for(int i = 0; i< NB_TILES_X  ;i++){
        if(hasNeighbourChanged(i,j)){
          x=i * TILE_W;
          y=j * TILE_H;
          who = omp_get_thread_num();   
          res = do_tile_vec(x, y , TILE_W, TILE_H, who);
          threadChange[who] |= res;
          *next_mapAddr(i,j)=res;
        }
      }
    }
    t2           = what_time_is_it ();
    cpu_duration = t2 - t1;

   
    
    // CPU waiting for the GPU to finish
    //clFinish (queue);
    //  GPU monitoring 
    gpu_duration = ocl_monitor (kernel_event, 0, cpu_y_part, global[0],
                                global[1], TASK_TYPE_COMPUTE);

    clReleaseEvent (kernel_event);
    gpu_accumulated_lines += gpu_y_part;
    clFinish (queue);
    //Load balancing
    
    if ( gpu_duration && cpu_duration) {
      ////CPU is stealing load
      if (much_greater_than (gpu_duration, cpu_duration) &&
                            gpu_y_part > TILE_H*2) {

        cpu_steals_load();
        gpu_y_part -= TILE_H;
        cpu_y_part += TILE_H;
        global[1] = gpu_y_part;
      }
      //// GPU is stealing load
      else if (much_greater_than (cpu_duration, gpu_duration) &&
                                  cpu_y_part > TILE_H*2) {
        gpu_steals_load();
        cpu_y_part -= TILE_H;
        gpu_y_part += TILE_H;
        global[1] = gpu_y_part;
      }else{
        do_statusQuo();        
      }
    }
    else{
      do_statusQuo();
    }
    
    // Retrieves the GpuChangeBuffer
    err = clEnqueueReadBuffer(queue, changeBuffer, CL_TRUE, 0, sizeof(unsigned),
          &gpuChange, 0, NULL, &kernel_event);
    check (err, "Failed to Read the buffer");    

    //  clearing current map for next itteration
    deleteCurrentBtmp();
    setGPUborderBtmp();

    swap_buffers();
    swap_tables();

    if(!gpuChange && !hasAnyTileChanged())
      return it;
  }

  PRINT_DEBUG ('u', "In average, GPU took %.1f%% of the lines\n",
                 (float)gpu_accumulated_lines * 100 / (DIM * nb_iter));

  return 0;
}

///////////////////////////// Initial configs

void life_draw_guns (void);

static inline void set_cell (int y, int x)
{
  if(ENABLE_BITCELL==2)
    setcurBitCellRow(x,y,1);
  else
    cur_table (y, x) = 1;
  if (opencl_used){
      *((cell_t*)image +(y+1) * (DIM+VEC_SIZE*2) + (x + VEC_SIZE)) = 1;
  }
}

static inline int get_cell (int y, int x)
{
  if(ENABLE_BITCELL==2)
    return getBitCellRow(x,y);
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