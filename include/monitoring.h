#ifndef MONITORING_IS_DEF
#define MONITORING_IS_DEF

#include "gmonitor.h"
#include "time_macros.h"
#include "trace_record.h"

#ifdef ENABLE_MONITORING

static inline void monitoring_declare_task_ids (char *task_ids[])
{
  trace_record_declare_task_ids (task_ids);
}

#ifdef ENABLE_SDL

#define monitoring_start_iteration()                                           \
  ({                                                                           \
    long t = 0;                                                                \
    do {                                                                       \
      if (do_gmonitor | do_trace) {                                            \
        t = what_time_is_it ();                                                \
        gmonitor_start_iteration (t);                                          \
        trace_record_start_iteration (t);                                      \
      }                                                                        \
    } while (0);                                                               \
    t;                                                                         \
  })

#define monitoring_end_iteration()                                             \
  ({                                                                           \
    long t = 0;                                                                \
    do {                                                                       \
      if (do_gmonitor | do_trace) {                                            \
        t = what_time_is_it ();                                                \
        gmonitor_end_iteration (t);                                            \
        trace_record_end_iteration (t);                                        \
      }                                                                        \
    } while (0);                                                               \
    t;                                                                         \
  })

#define monitoring_start_tile(c)                                               \
  do {                                                                         \
    if (do_gmonitor | do_trace) {                                              \
      long t = what_time_is_it ();                                             \
      gmonitor_start_tile (t, (c));                                            \
      trace_record_start_tile (t, (c));                                        \
    }                                                                          \
  } while (0)

#define monitoring_end_tile(x, y, w, h, c)                                     \
  do {                                                                         \
    if (do_gmonitor | do_trace) {                                              \
      long t = what_time_is_it ();                                             \
      gmonitor_end_tile (t, (c), (x), (y), (w), (h));                          \
      trace_record_end_tile (t, (c), (x), (y), (w), (h), TASK_TYPE_COMPUTE,    \
                             0);                                               \
    }                                                                          \
  } while (0)

#define monitoring_end_tile_id(x, y, w, h, c, id)                              \
  do {                                                                         \
    if (do_gmonitor | do_trace) {                                              \
      long t = what_time_is_it ();                                             \
      gmonitor_end_tile (t, (c), (x), (y), (w), (h));                          \
      trace_record_end_tile (t, (c), (x), (y), (w), (h), TASK_TYPE_COMPUTE,    \
                             (id) + 1);                                        \
    }                                                                          \
  } while (0)

#define monitoring_gpu_tile(x, y, w, h, c, s, e, tt)                           \
  do {                                                                         \
    if (do_gmonitor | do_trace) {                                              \
      if ((tt) == TASK_TYPE_COMPUTE)                                           \
        gmonitor_start_tile ((s), (c));                                        \
      trace_record_start_tile ((s), (c));                                      \
      if ((tt) == TASK_TYPE_COMPUTE)                                           \
        gmonitor_end_tile ((e), (c), (x), (y), (w), (h));                      \
      trace_record_end_tile ((e), (c), (x), (y), (w), (h), (tt), 0);           \
    }                                                                          \
  } while (0)

#else // no SDL

#define monitoring_start_iteration()                                           \
  ({                                                                           \
    long t = 0;                                                                \
    do {                                                                       \
      if (do_trace) {                                                          \
        t = what_time_is_it ();                                                \
        trace_record_start_iteration (t);                                      \
      }                                                                        \
    } while (0);                                                               \
    t;                                                                         \
  })

#define monitoring_end_iteration()                                             \
  ({                                                                           \
    long t = 0;                                                                \
    do {                                                                       \
      if (do_trace) {                                                          \
        t = what_time_is_it ();                                                \
        trace_record_end_iteration (t);                                        \
      }                                                                        \
    } while (0);                                                               \
    t;                                                                         \
  })

#define monitoring_start_tile(c)                                               \
  do {                                                                         \
    if (do_trace) {                                                            \
      long t = what_time_is_it ();                                             \
      trace_record_start_tile (t, (c));                                        \
    }                                                                          \
  } while (0)

#define monitoring_end_tile(x, y, w, h, c)                                     \
  do {                                                                         \
    if (do_trace) {                                                            \
      long t = what_time_is_it ();                                             \
      trace_record_end_tile (t, (c), (x), (y), (w), (h), TASK_TYPE_COMPUTE,    \
                             0);                                               \
    }                                                                          \
  } while (0)

#define monitoring_end_tile_id(x, y, w, h, c, id)                              \
  do {                                                                         \
    if (do_trace) {                                                            \
      long t = what_time_is_it ();                                             \
      trace_record_end_tile (t, (c), (x), (y), (w), (h), TASK_TYPE_COMPUTE,    \
                             (id) + 1);                                        \
    }                                                                          \
  } while (0)

#define monitoring_gpu_tile(x, y, w, h, c, s, e, tt)                           \
  do {                                                                         \
    if (do_trace) {                                                            \
      trace_record_start_tile ((s), (c));                                      \
      trace_record_end_tile ((e), (c), (x), (y), (w), (h), (tt), 0);           \
    }                                                                          \
  } while (0)

#endif

#else

#define monitoring_declare_task_ids (task_ids) (void) 0
#define monitoring_start_iteration() (void)0
#define monitoring_end_iteration() (void)0
#define monitoring_start_tile(c) (void)0
#define monitoring_end_tile(x, y, w, h, c) (void)0
#define monitoring_end_tile_id(x, y, w, h, c, id) (void)0
#define monitoring_gpu_tile(x, y, w, h, c, s, e, tt) (void)0

#endif

#endif
