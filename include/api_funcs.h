#ifndef API_FUNCS_IS_DEF
#define API_FUNCS_IS_DEF

typedef enum {
    VEC_TYPE_CHAR,
    VEC_TYPE_INT,
    VEC_TYPE_FLOAT,
    VEC_TYPE_DOUBLE
} vec_type_t;

typedef enum {
    DIR_HORIZONTAL,
    DIR_VERTICAL
} direction_t;

unsigned easypap_requested_number_of_threads (void);
unsigned easypap_number_of_cores (void);
unsigned easypap_number_of_gpus (void);
int easypap_mpi_rank (void);
int easypap_mpi_size (void);
void easypap_check_mpi (void);
void easypap_check_vectorization (vec_type_t vec_type, direction_t dir);
int easypap_proc_is_master (void);


#endif