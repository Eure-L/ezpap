#include <omp.h>
#include <stdbool.h>

struct task_t{
  int tile_x;
  int tile_y;
};typedef struct task_t task;

struct taskStack_t{
  task * tasks;
  int stackSize;
  int nbTasks;
};typedef struct taskStack_t taskStack;

/**
 * @brief Create a Task object
 * 
 * @param x 
 * @param y 
 * @return task 
 */
task createTask(int x, int y);

/**
 * @brief 
 * 
 * @param t 
 */
void printTask(task t);

/**
 * @brief Initializer for the structure taskstack
 * 
 * @return stack
 */
taskStack taskStackInit(void);

/**
 * @brief 
 * 
 * @param stack 
 */
void taskStackDelete(taskStack * stack);

/**
 * @brief Stack a given task, uses pointer arithmetics
 * 
 * 
 * @param stack 
 * @param taskToStack 
 * @return taskStack* 
 */
void addTask(taskStack * stack, task taskToStack);

/**
 * @brief Pops a given Stack returning a task
 * 
 * @param stack 
 * @return task 
 */
void delStack(taskStack * stack);

/**
 * @brief prints the stacks
 * For debugging purpose
 * 
 * @param stack 
 */
void printTaskStack(taskStack * stack);

/**
 * @brief The four next functions are used to determine if a given pixel
 * is on the border (accordingly to the function name) of a tile
 * 
 * @param x 
 * @param y 
 * @return true 
 * @return false 
 */
bool isOnLeft(int x,int y);

bool isOnRight(int x,int y);

bool isOnTop(int x,int y);

bool isOnBottom(int x,int y);

/**
 * @brief Initializes the stack of tasks data structure that the lazy
 * compute algorithm will use
 *                    
 *  [task1;task2;task3...;task(n)]          ===>            [task3;task54;task69]
 *      current stack of tasks       after one computation        future stack of tasks
 * 
 * It is composed of two stacks of tasks
 * At the Start/first itteration, we set the stack of tasks completly full of tasks
 * then after an other itteration, thanks to the lazy algorithm that determines which 
 * tiles to compute we'll obtain a new set of tasks necessary to compute that 
 * well be much smaller than the first one.
 * 
 * These two stacks are used for all the itterations alternating between 
 * the current stack of tasks and the next one (to save space and mem allocation time)
 * 
 * 
 * @param lock 
 * @param curr_tasks 
 * @return taskStack* 
 */
taskStack * initStacks( omp_lock_t * lock, int curr_tasks);

/**
 * @brief Initializes the Bitmaps data structures that we'll use the lazu algorithm 
 * to work on; there is two BitMaps Arrays
 * 
 * like the previous one
 * 
 * Here's an example, a workload of dimention 5 * 5 tiles
 * 
 *  1 1 1 1 1                     1 1 0 1 1
 *  1 1 1 1 1     one itteration  1 0 0 0 1 
 *  1 1 1 1 1         ====>       0 0 0 0 0
 *  1 1 1 1 1                     1 0 0 0 1  
 *  1 1 1 1 1                     1 1 0 1 1
 *  current                         next
 * 
 * 1: tiles to be computed
 * 0: tiles not to be
 * 
 * Here again it keeps the tiles that change in the current itteration
 * to be computed in the next itteration
 * 
 * 
 * @param lock 
 * @param curr_tasks 
 * @return char*** 
 */
char *** initBtmptls(omp_lock_t * lock,int curr_tasks);