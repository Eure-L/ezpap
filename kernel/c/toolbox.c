#include "easypap.h"
#include "rle_lexer.h"
#include "toolbox.h"

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

void printTask(task t){
  printf("(%d,%d) - ",t.tile_x,t.tile_y);
}

taskStack taskStackInit(void){
  taskStack stack;
  stack.tasks = (task *) malloc(((DIM*DIM)/(NB_TILES_X*NB_TILES_Y))*sizeof(task)); //Don't forget to Free
  stack.tasks[0].tile_x=-1;
  stack.tasks[0].tile_y=-1;
  stack.stackSize=((DIM*DIM)/(NB_TILES_X*NB_TILES_Y));
  stack.nbTasks=0;
  return stack;
}

void taskStackDelete(taskStack * stack){
  free(stack->tasks);
}

void addTask(taskStack * stack, task taskToStack){
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
  //printf("adding task\n");

}

void delStack(taskStack * stack){
  if(stack->nbTasks<=0)
    printf("Can't pop it more\n");
  //task null = createTask(0,0);
  stack->nbTasks=0; 
}

void printTaskStack(taskStack * stack){
    
    for(int ptr=0;ptr<(*stack).nbTasks;ptr++){
      printTask((*stack).tasks[ptr]);
    }
    printf("\n");
}

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

taskStack * initStacks(omp_lock_t * lock, int curr_tasks){   
    taskStack * tasks = (taskStack *) malloc (2 * sizeof(taskStack));
    tasks[0]=taskStackInit(); // one stack of tasks for the threads to pick
    tasks[1]=taskStackInit(); // an other one for the threads to foresee the load in the next itteration 
    omp_init_lock(lock);
    //We start by filling all the tiles in the stack of anticipated load 
    task startTask;

    for(int i=0;i<NB_TILES_X;i++){
      for(int j=0;j<NB_TILES_Y;j++){
        startTask = createTask(i* TILE_W,j* TILE_H);
        addTask(&tasks[curr_tasks],startTask);
      }
    }
    return tasks;
}

char *** initBtmptls(omp_lock_t * lock,int curr_tasks){
  
  char *** map = (char ***) malloc (2 * sizeof(char **));
  omp_init_lock(lock);
  if(map == NULL){
    printf("map pointer NULL\n");
    exit(EXIT_FAILURE);
  }     
         
  map[0] = (char **) malloc (NB_TILES_X * sizeof(char *));            
  map[1] = (char **) malloc (NB_TILES_X * sizeof(char *));
  if(map[0] == NULL || map[1]==NULL){
    printf("map second pointer is NULL\n");
    exit(EXIT_FAILURE);
  }     

  int idMap;
  for (int i=0;i<2*NB_TILES_X;i++){
    idMap = i/NB_TILES_X;                             
    map[idMap][i]=(char *) malloc (NB_TILES_Y * sizeof(char));
    if(map[idMap][i] == NULL){
      printf("map third pointer is NULL : idMap = %d & i = %d\n",idMap,i);
      exit(EXIT_FAILURE);
    }

    for(int j=0;j<NB_TILES_Y;j++){
      if(idMap==curr_tasks)
        map[idMap][i][j] = 1;
      else
        map[idMap][i][j] = 0;
    }
  }
  return map;
}


