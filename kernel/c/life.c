#include "easypap.h"
#include "rle_lexer.h"
#include "toolbox.h"

#include <omp.h>
#include <stdbool.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>


static unsigned color = 0xFFFF00FF; // Living cells have the yellow color

typedef char cell_t;

static cell_t *restrict _table = NULL, *restrict _alternate_table = NULL;

taskStack * tasks;
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

// This kernel does not directly work on cur_img/next_img.
// Instead, we use 2D arrays of boolean values, not colors
#define cur_table(y, x) (*table_cell (_table, (y), (x)))
#define next_table(y, x) (*table_cell (_alternate_table, (y), (x)))


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

void addCreateTask(int x, int y){
        task futureTask = createTask(x,y);
        omp_set_lock(writelock);
        addTask(tasks+next_tasks,futureTask);
        omp_unset_lock(writelock);
}

static int lazy_compute_new_state (int y, int x)
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
      bool left = isOnLeft(x,y);
      bool right = isOnRight(x,y);
      bool top = isOnTop(x,y);
      bool bottom = isOnBottom(x,y);

      if(left){
        addCreateTask(x-TILE_W,(y/TILE_H)*TILE_H);    
        if(top)
          addCreateTask(x-TILE_W,y-TILE_H);
        else if(bottom)
          addCreateTask(x-TILE_W,y+1);
      }
      else if (right){
        addCreateTask(x+1,(y/TILE_H)*TILE_H);  
        if(top)
          addCreateTask(x+1,y-TILE_H);
        else if(bottom)
          addCreateTask(x+1,y+1);   
      }
      if(top){
        addCreateTask((x/TILE_W)*TILE_W,y-TILE_H);   
      }
      else if (bottom){
        addCreateTask((x/TILE_W)*TILE_W,y+1);
      }
    }
  }

  return change;
}

static int lazy_do_tile_reg (int x, int y, int width, int height)
{
  int change = 0;

  for (int i = y; i < y + height; i++){
    for (int j = x; j < x + width; j++){
      change |= lazy_compute_new_state (i, j);     
    }
  }
  return change;
}

static int lazy_do_tile (int x, int y, int width, int height, int who)
{
  int r;

  monitoring_start_tile (who);

  r = lazy_do_tile_reg (x, y, width, height);

  monitoring_end_tile (x, y, width, height, who);

  return r;
}


//////////////////////// first lazy version
unsigned life_compute_lazy(unsigned nb_iter)
{
  unsigned  res = 0;

  if(init){ // init section of the data structures
    writelock = (omp_lock_t *) malloc(sizeof(omp_lock_t));
    tasks = initStacks(writelock,curr_tasks);
    init=false;
  }
  
  //Main loop
  for (unsigned it = 1; it <= nb_iter; it++) {
    curr_tasks = switcher;
    next_tasks = !switcher;
    unsigned change = 0;
    unsigned nbTsk  =tasks[curr_tasks].nbTasks;

    //Distribution of the task creation process amongst every threads (might not be the best)
    #pragma omp parallel for schedule(dynamic)
    for (unsigned taskNum = 0; taskNum < nbTsk; taskNum++){
      int x = tasks[curr_tasks].tasks[taskNum].tile_x ;
      int y = tasks[curr_tasks].tasks[taskNum].tile_y ;
      
      #pragma omp task
       {
        unsigned tileChange = 0;
        // do_tile compute the inner pixels of the tile, not bothering
        // if we should compute bordering tiles in the next itteration
        //
        // lazy_do_tile is used for the outer pixels of the tile, adding for the next
        // itteration the borduring tile if the current pixel has changed state
        tileChange |= do_tile (x+1, y+1, TILE_W-2, TILE_H-2, omp_get_thread_num());//inner tile
        tileChange |= lazy_do_tile(x,y,TILE_W,1,omp_get_thread_num());//top
        tileChange |= lazy_do_tile(x,y+TILE_H-1,TILE_W,1,omp_get_thread_num());//bot
        tileChange |= lazy_do_tile(x,y+1,1,TILE_H-2,omp_get_thread_num());//left
        tileChange |= lazy_do_tile(x+TILE_W-1,y+1,1,TILE_H-2,omp_get_thread_num());//right
        
        if(tileChange){       //if the tile change, we'll compute it again in the
          addCreateTask(x,y); //next itteration
        }
        change |= tileChange;
      }
    }

    delStack(tasks+curr_tasks);   
    swap_tables ();

    if (!change) { // we stop when all cells are stable
      res = it;
      printf("there's no future tasks\n");
      break;
    }
    switcher = ! switcher;
  }
  return res;
}

static int btmp_compute_new_state (int y, int x)
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
      bool left = isOnLeft(x,y);
      bool right = isOnRight(x,y);
      bool top = isOnTop(x,y);
      bool bottom = isOnBottom(x,y);

      if(left){
        addTaskBtmp((x-TILE_W)/TILE_W,(y/TILE_H),bitMapTls+nextTable*NB_TILES_TOT);    
        if(top)
          addTaskBtmp((x-TILE_W)/TILE_W,(y-TILE_H)/TILE_H,bitMapTls+nextTable*NB_TILES_TOT);
        else if(bottom)
          addTaskBtmp((x-TILE_W)/TILE_W,(y+1)/TILE_H,bitMapTls+nextTable*NB_TILES_TOT);
      }
      else if (right){
        addTaskBtmp((x+1)/TILE_W,(y/TILE_H),bitMapTls+nextTable*NB_TILES_TOT);  
        if(top)
          addTaskBtmp((x+1)/TILE_W,(y-TILE_H)/TILE_H,bitMapTls+nextTable*NB_TILES_TOT);
        else if(bottom)
          addTaskBtmp((x+1)/TILE_W,(y+1)/TILE_H,bitMapTls+nextTable*NB_TILES_TOT);   
      }
      if(top){
        addTaskBtmp((x/TILE_W),(y-TILE_H)/TILE_H,bitMapTls+nextTable*NB_TILES_TOT);   
      }
      else if (bottom){
        addTaskBtmp((x/TILE_W),(y+1)/TILE_H,bitMapTls+nextTable*NB_TILES_TOT);
      }
    }
  }

  return change;
}

static int btmp_do_tile_reg (int x, int y, int width, int height)
{
  int change = 0;

  for (int i = y; i < y + height; i++){
    for (int j = x; j < x + width; j++){
      change |= btmp_compute_new_state (i, j);     
    }
  }
  return change;
}

static int btmp_do_tile (int x, int y, int width, int height, int who)
{
  int r;

  monitoring_start_tile (who);

  r = btmp_do_tile_reg (x, y, width, height);

  monitoring_end_tile (x, y, width, height, who);

  return r;
}

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
            tileChange |= btmp_do_tile(x,y,TILE_W,1,omp_get_thread_num());//top
            tileChange |= btmp_do_tile(x,y+TILE_H-1,TILE_W,1,omp_get_thread_num());//bot
            tileChange |= btmp_do_tile(x,y+1,1,TILE_H-2,omp_get_thread_num());//left
            tileChange |= btmp_do_tile(x+TILE_W-1,y+1,1,TILE_H-2,omp_get_thread_num());//right
            
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

#define right 1
#define left 2
#define top 4
#define topright 5
#define topleft 6
#define toprightleft 7
#define bot 8
#define botright 9
#define botleft 10
#define botrightleft 11
#define righttopbot 13
#define lefttopbot 14
#define rightlefttopbot 15

// rightleft and topbot are forbidden

static int btmp2_compute_new_state (int y, int x,unsigned tilepos)
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
      unsigned mask = 0;
      mask |= isOnRight(x,y) ? 1 : 0 ;
      mask |= isOnLeft(x,y) ? 2 : 0 ;
      mask |= isOnTop(x,y) ? 4 : 0 ;
      mask |= isOnBottom(x,y) ? 8 : 0;
      switch (mask)
      {
        case right: addTaskBtmp((x+1)/TILE_W,(y/TILE_H),bitMapTls+nextTable*NB_TILES_TOT); break;
        case left: addTaskBtmp((x-TILE_W)/TILE_W,(y/TILE_H),bitMapTls+nextTable*NB_TILES_TOT); break; 
        case top: addTaskBtmp((x/TILE_W),(y-TILE_H)/TILE_H,bitMapTls+nextTable*NB_TILES_TOT); break; 
        case bot: addTaskBtmp((x/TILE_W),(y+1)/TILE_H,bitMapTls+nextTable*NB_TILES_TOT); break; 
        case topright: addTaskBtmp((x+1)/TILE_W,(y-TILE_H)/TILE_H,bitMapTls+nextTable*NB_TILES_TOT) ;break; 
        case topleft: addTaskBtmp((x-TILE_W)/TILE_W,(y-TILE_H)/TILE_H,bitMapTls+nextTable*NB_TILES_TOT); break;
        case botright: addTaskBtmp((x+1)/TILE_W,(y+1)/TILE_H,bitMapTls+nextTable*NB_TILES_TOT); break;
        case botleft: addTaskBtmp((x-TILE_W)/TILE_W,(y+1)/TILE_H,bitMapTls+nextTable*NB_TILES_TOT); break;
        default:
          break;
      }
    }
  }

  return change;
}

static int btmp2_do_tile_reg (int x, int y, int width, int height,unsigned tilepos)
{
  int change = 0;

  for (int i = y; i < y + height; i++){
    for (int j = x; j < x + width; j++){
      change |= btmp2_compute_new_state (i, j,tilepos);     
    }
  }
  return change;
}

static int btmp2_do_tile (int x, int y, int width, int height, int who,unsigned tilePos)
{
  int r;

  monitoring_start_tile (who);

  r = btmp2_do_tile_reg (x, y, width, height,tilePos);

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


bool do_tileLauncher(unsigned tilePos,unsigned x,unsigned y){
  bool tileChange = False;
  tileChange |= btmp2_do_tile(x,y,TILE_W,1,omp_get_thread_num(),tilePos);//top
  tileChange |= btmp2_do_tile(x,y+1,1,TILE_H-2,omp_get_thread_num(),tilePos);//left
  tileChange |= do_tile_nocheck (x +1, y +1, TILE_W-2, TILE_H-2, omp_get_thread_num());//inner tile
  tileChange |= btmp2_do_tile(x+TILE_W-1,y+1,1,TILE_H-2,omp_get_thread_num(),tilePos);//right
  tileChange |= btmp2_do_tile(x,y+TILE_H-1,TILE_W,1,omp_get_thread_num(),tilePos);//bot
  
  return tileChange;
}

//////////////////////// BitMap2 lazy Version ;
// ./run -k life -s 128 -ts 32 -v lazybtmp2 -m
// ./run -k life -a random -s 2048 -ts 32 -v lazybtmp2 -m
// ./run -k life -a otca_off -s 2196 -ts 32 -v lazybtmp2 -m
unsigned life_compute_lazybtmp2 (unsigned nb_iter){
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
    #pragma omp parallel for  schedule(static)
    for(int i = 0; i< NB_TILES_X;i++){
      for(int j = 0; j< NB_TILES_Y;j++){
        if(*(bitMapTls+curTable*NB_TILES_TOT+(j*NB_TILES_X)+i)==1){
            unsigned x=i * TILE_W;
            unsigned y=j * TILE_H;
            unsigned tilePos = tilePosition(i,j);
            unsigned tileChange = do_tileLauncher(tilePos,x,y);      
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

unsigned life_compute_vec (unsigned nb_iter){
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
    for(int i = 0; i< NB_TILES_X;i++){
      for(int j = 0; j< NB_TILES_Y;j++){
        if(*(bitMapTls+curTable*NB_TILES_TOT+(j*NB_TILES_X)+i)==1){
            unsigned x=i * TILE_W;
            unsigned y=j * TILE_H;
            unsigned tilePos = tilePosition(i,j);
            unsigned tileChange = do_tileLauncher(tilePos,x,y);      
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
