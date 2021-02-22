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
 * @brief 
 * 
 * @param stack 
 */
void printTaskStack(taskStack * stack);

bool isOnLeft(int x,int y);

bool isOnRight(int x,int y);

bool isOnTop(int x,int y);

bool isOnBottom(int x,int y);

taskStack * initStacks( omp_lock_t * lock, int curr_tasks);

void initBtmptls(char *** map);