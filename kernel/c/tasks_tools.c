#include "easypap.h"
#include "rle_lexer.h"
#include "tasks_tools.h"

#include <omp.h>
#include <stdbool.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>


task createTask(int x, int y){
    task newTask;
    newTask.tile_x=x;
    newTask.tile_y=y;
    return newTask;
}

taskStack taskStackInit(void){
  taskStack stack;
  stack.tasks = (task *) malloc(DIM*sizeof(task)); //Don't forget to Free
  stack.tasks[0].tile_x=-1;
  stack.tasks[0].tile_y=-1;
  stack.stackSize=DIM;
  stack.nbTasks=0;
  return stack;
}

void taskStackDelete(taskStack * stack){
  free(stack->tasks);
}

void stacking(taskStack * stack, task taskToStack){
  
  if(stack->nbTasks==stack->stackSize){
    stack->stackSize = stack->stackSize * 2;
    stack->tasks = realloc(stack->tasks,stack->stackSize*sizeof(task));
  }
  for(int i=0;i<stack->nbTasks;i++){
    if(stack->tasks[i].tile_x==taskToStack.tile_x&&stack->tasks[i].tile_y==taskToStack.tile_y)
      return;
  }
  stack->nbTasks += 1;
  stack->tasks[stack->nbTasks-1].tile_x = taskToStack.tile_x;
  stack->tasks[stack->nbTasks-1].tile_y = taskToStack.tile_y;
}

task poping(taskStack * stack){
  if(stack->nbTasks<=0)
    printf("Can't pop it more\n");
  stack->nbTasks--;
  return stack->tasks[stack->nbTasks];
}

void printTaskStack(taskStack * stack){
    int x = DIM/TILE_W;
    int y = DIM/TILE_H;
    bool table[x][y];
    for(int i = 0 ; i < x; i++){
        for(int j=0 ; j < y ; j++){
            table[i][j]=false;
        }
    }
    for(int ptr=0;ptr<(*stack).nbTasks;ptr++){
        x = ((*stack).tasks->tile_x)/TILE_W;
        y = ((*stack).tasks->tile_y)/TILE_H;
        table[x][y] = true;
    }
    for(int i = 0 ; i < x; i++){
        for(int j=0 ; j < y ; j++){
            printf("%d ",table[i][j]);
        }
        printf("\n");
    }
}

