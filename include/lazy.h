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
 * @brief Initializer for the structure taskstack
 * 
 * @return stack
 */
taskStack taskStackInit(void);

/**
 * @brief Stack a given task, uses pointer arithmetics
 * 
 * 
 * @param stack 
 * @param taskToStack 
 * @return taskStack* 
 */
void stacking(taskStack stack, task taskToStack);

/**
 * @brief Pops a given Stack returning a task
 * 
 * @param stack 
 * @return task 
 */
task poping(taskStack stack);

