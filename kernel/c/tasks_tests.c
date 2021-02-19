#include "easypap.h"
#include "rle_lexer.h"
#include "tasks_tools.h"

#include <omp.h>
#include <stdbool.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>


bool test_create(){
    task tache = createTask(1,2);
    if(tache.tile_x=!1 && tache.tile_y!=2)
        return false;
}

int main(void){



}