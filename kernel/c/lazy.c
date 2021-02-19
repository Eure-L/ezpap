#include "easypap.h"
#include "rle_lexer.h"
#include "lazy.h"

#include <omp.h>
#include <stdbool.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>


/**
 * @brief Initializer for the structure taskstack
 * 
 * @return stack
 */
taskStack taskStackInit(void){
  taskStack stack;
  stack.tasks = (task *) malloc(DIM*sizeof(task)); //Don't forget to Free
  stack.tasks[0].tile_x=-1;
  stack.tasks[0].tile_y=-1;
  stack.stackSize=DIM;
  stack.nbTasks=0;

  return stack;
}

/**
 * @brief Stack a given task, uses pointer arithmetics
 * 
 * 
 * @param stack 
 * @param taskToStack 
 * @return taskStack* 
 */
void stacking(taskStack stack, task taskToStack){
  
  if(stack.nbTasks==stack.stackSize){
    stack.stackSize = stack.stackSize * 2;
    stack.tasks = realloc(stack.tasks,stack.stackSize*sizeof(task));
  }
  for(int i=0;i<stack.nbTasks;i++){
    if(stack.tasks[i].tile_x==taskToStack.tile_x&&stack.tasks[i].tile_y==taskToStack.tile_y)
      return;
  }
  stack.nbTasks += 1;
  stack.tasks[stack.nbTasks-1].tile_x = taskToStack.tile_x;
  stack.tasks[stack.nbTasks-1].tile_y = taskToStack.tile_y;
}
/**
 * @brief Pops a given Stack returning a task
 * 
 * @param stack 
 * @return task 
 */
task poping(taskStack stack){
  if(stack.nbTasks<=0)
    printf("Can't pop it more\n");
  stack.nbTasks--;
  return stack.tasks[stack.nbTasks];
}

