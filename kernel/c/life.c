
#include "easypap.h"
#include "rle_lexer.h"
#include "tasks_tools.h"

#include <omp.h>
#include <stdbool.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>


static unsigned color = 0xFFFF00FF; // Living cells have the yellow color

typedef unsigned cell_t;

static cell_t *restrict _table = NULL, *restrict _alternate_table = NULL;

taskStack * tasks;
bool init = true;
bool switcher = 1;
omp_lock_t writelock;

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
  for (unsigned it = 1; it <= nb_iter; it++) {
    

    int change = 0;

    monitoring_start_tile (0);

    for (int i = 0; i < DIM; i++)
      for (int j = 0; j < DIM; j++)
        change |= compute_new_state (i, j);

    monitoring_end_tile (0, 0, DIM, DIM, 0);

    //swap_tables ();

    if (!change)
      return it;
  }
  return 0;
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
unsigned life_compute_omp_tiled (unsigned nb_iter)
{

  unsigned res = 0;

  for (unsigned it = 1; it <= nb_iter; it++) {
    unsigned change = 0;
    #pragma omp parallel
    #pragma omp single
    for (int y = 0; y < DIM; y += TILE_H)
      for (int x = 0; x < DIM; x += TILE_W)
        #pragma omp task
        change |= do_tile (x, y, TILE_W, TILE_H, omp_get_thread_num());

    swap_tables ();

    if (!change) { // we stop when all cells are stable
      res = it;
      break;
    }
  }

  return res;
}

static int lazy_compute_new_state (int y, int x, taskStack * stack)
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
      if(x > 0 && x%TILE_W == 0 ){
        task futureTask = createTask(x-TILE_W,(y/TILE_H)*TILE_H);
        // printf("1# %d,%d#",x,y);
        // printTask(futureTask);
        omp_set_lock(&writelock);
        addTask(stack,futureTask);
        omp_unset_lock(&writelock);
      }
      else if (x<DIM-1 && (x%TILE_W == TILE_W-1)){
        task futureTask = createTask(x+1,(y/TILE_H)*TILE_H);
        // printf("2# %d,%d#",x,y);
        // printTask(futureTask);
        omp_set_lock(&writelock);
        addTask(stack,futureTask);
        omp_unset_lock(&writelock);
      }
      if(y>0 && y%TILE_H==0){
        task futureTask = createTask((x/TILE_W)*TILE_W,y-TILE_H);
        // printf("3# %d,%d#",x,y);
        // printTask(futureTask);
        omp_set_lock(&writelock);
        addTask(stack,futureTask);
        omp_unset_lock(&writelock);
      }
      else if (y<DIM-1 && (y%TILE_H == TILE_H-1)){
        task futureTask = createTask((x/TILE_W)*TILE_W,y+1);
        // printf("4# %d,%d#",x,y);
        // printTask(futureTask);
        omp_set_lock(&writelock);
        addTask(stack,futureTask);
        omp_unset_lock(&writelock);
      }
    }
  }

  return change;
}

static int lazy_do_tile_reg (int x, int y, int width, int height, taskStack * stack)
{
  int change = 0;

  for (int i = y; i < y + height; i++){
    for (int j = x; j < x + width; j++){
      change |= lazy_compute_new_state (i, j,stack);     
    }
  }
  return change;
}

static int lazy_do_tile (int x, int y, int width, int height, int who,taskStack * stack)
{
  int r;

  monitoring_start_tile (who);

  r = lazy_do_tile_reg (x, y, width, height,stack);

  monitoring_end_tile (x, y, width, height, who);

  return r;
}

unsigned lazy_life_compute_omp_tiled (unsigned nb_iter,taskStack * stack)
{
  unsigned res = 0;
  for (unsigned it = 1; it <= nb_iter; it++) {
    unsigned change = 0;
    #pragma omp parallel
    #pragma omp single
    for (int y = 0; y < DIM; y += TILE_H)
      for (int x = 0; x < DIM; x += TILE_W)
        #pragma omp task
        change |= lazy_do_tile (x, y, TILE_W, TILE_H, omp_get_thread_num(),stack);

    swap_tables ();

    if (!change) { // we stop when all cells are stable
      res = it;
      break;
    }
  }

  return res;
}

//////////////////////// lazy version
unsigned life_compute_lazy (unsigned nb_iter)
{
  //printf("%d\n",it);
  unsigned curr_tasks = 1;
  unsigned next_tasks = 0;
  unsigned  res = 0;

  if(init){
    tasks = (taskStack *) malloc (2 * sizeof(taskStack));
    tasks[0]=taskStackInit(); // one stack of tasks for the threads to pick
    tasks[1]=taskStackInit(); // an other one for the threads to foresee the load in the next itteration 
    omp_init_lock(&writelock);

    //We start by filling all the tiles in the stack of anticipated load 
    task startTask;
    for(int i=0;i<NB_TILES_X;i++){
      for(int j=0;j<NB_TILES_Y;j++){
        startTask = createTask(i* TILE_W,j* TILE_H);
        addTask(&tasks[curr_tasks],startTask);
      }
    }
    init=false;
  }
  for (unsigned it = 1; it <= nb_iter; it++) {
    curr_tasks = switcher;
    next_tasks = !switcher;
    unsigned change = 0;
    unsigned nbTsk  =tasks[curr_tasks].nbTasks;

    #pragma omp parallel for schedule(dynamic)
    //#pragma omp single
    for (unsigned taskNum = 0; taskNum < nbTsk; taskNum++){

      int x = tasks[curr_tasks].tasks[taskNum].tile_x ;
      int y = tasks[curr_tasks].tasks[taskNum].tile_y ;
      #pragma omp task
       {
        bool tileChange = lazy_do_tile (x, y, TILE_W, TILE_H, omp_get_thread_num(),&tasks[next_tasks]);
        change |= tileChange;
        if(tileChange){
          omp_set_lock(&writelock);
          addTask(&tasks[next_tasks],createTask(x,y));
          omp_unset_lock(&writelock);

        }
      }
      //#pragma omp taskwait
    }

    delStack(tasks+curr_tasks);
    
    swap_tables ();

    if (tasks[next_tasks].nbTasks == 0) { // we stop when all cells are stable
      res = it;
      printf("there's no future tasks\n");
      omp_destroy_lock(&writelock);
      
      taskStackDelete(&tasks[0]);
      taskStackDelete(&tasks[1]);
      free(tasks);
      break;
    }
    switcher = ! switcher;
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
