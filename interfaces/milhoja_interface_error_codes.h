#if 0
--------------------------------------------------------------------------------
Both the Milhoja C and Fortran interfaces can report errors.  Therefore, the
contents of this file shall be written such that the file can be included and its
contents used in both C++ and Fortran codes.  In this way, we can include all
error codes in this single file.

Each macro shall be associated with a unique non-zero error code.  In the
interest of maintainablity and backward compatibility, once an error macro is
added to the file, it shall not be removed nor renamed.  Similary, the error
code associated with a macro shall not be changed and a function shall not use a
different macro unless it was using the incorrect macro originally.  In doing
so, error codes displayed during program execution shall not depend on the
version of the sofware that was used during execution.

TODO: Allow for macro renaming?  If the rename is propagated correctly to
all functions that used the original name, then the change should be transparent
to calling code.
TODO: The above should be in the developer's guide rather than here.
--------------------------------------------------------------------------------
#endif

#if 0
--------------------------------------------------------------------------------
General-use error codes
--------------------------------------------------------------------------------
#endif
#define ERROR_N_THREADS_NEGATIVE               1
#define ERROR_LEVEL_NEGATIVE                   2
#define ERROR_STEP_NEGATIVE                    3
#define ERROR_UNABLE_TO_GET_LIMITS             4
#define ERROR_UNABLE_TO_GET_POINTER            5

#if 0
--------------------------------------------------------------------------------
Runtime-specific error codes
--------------------------------------------------------------------------------
#endif
#define ERROR_UNABLE_TO_INIT_RUNTIME         100
#define ERROR_UNABLE_TO_FINALIZE_RUNTIME     101
#define ERROR_N_THREAD_TEAMS_NEGATIVE        102
#define ERROR_N_THREADS_PER_TEAM_NEGATIVE    103
#define ERROR_N_STREAMS_NEGATIVE             104
#define ERROR_UNABLE_TO_EXECUTE_TASKS        105
#define ERROR_N_DISTRIBUTOR_THREADS_NEGATIVE 106
#define ERROR_N_TEAM_THREADS_NEGATIVE        107

#if 0
--------------------------------------------------------------------------------
Grid-specific error codes
--------------------------------------------------------------------------------
#endif
#define ERROR_UNABLE_TO_INIT_GRID            200
#define ERROR_UNABLE_TO_FINALIZE_GRID        201
#define ERROR_UNABLE_TO_INIT_DOMAIN          202
#define ERROR_UNABLE_TO_GET_BOUNDS           203
#define ERROR_UNABLE_TO_GET_LEVEL            204
#define ERROR_UNABLE_TO_GET_DELTAS           205
#define ERROR_UNABLE_TO_WRITE_PLOTFILE       206

