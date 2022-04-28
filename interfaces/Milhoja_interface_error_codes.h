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
TODO: The above should be in the developers guide rather than here.
--------------------------------------------------------------------------------
#endif

#if 0
--------------------------------------------------------------------------------
Indicate to calling code that execution was successful
--------------------------------------------------------------------------------
#endif
#define MILHOJA_SUCCESS                                0

#if 0
--------------------------------------------------------------------------------
General-use error codes
--------------------------------------------------------------------------------
#endif
#define MILHOJA_ERROR_N_THREADS_NEGATIVE               1
#define MILHOJA_ERROR_INVALID_LEVEL                    2
#define MILHOJA_ERROR_STEP_NEGATIVE                    3
#define MILHOJA_ERROR_UNABLE_TO_GET_LIMITS             4
#define MILHOJA_ERROR_UNABLE_TO_GET_POINTER            5
#define MILHOJA_ERROR_POINTER_IS_NULL                  6
#define MILHOJA_ERROR_POINTER_NOT_NULL                 7
#define MILHOJA_ERROR_NEGATIVE_VALUE_FOR_UINT          8

#if 0
--------------------------------------------------------------------------------
Type-specific error codes
--------------------------------------------------------------------------------
#endif
#define MILHOJA_ERROR_BAD_I32_SIZE                   100
#define MILHOJA_ERROR_BAD_I64_SIZE                   101
#define MILHOJA_ERROR_INT_SIZE_MISMATCH              102
#define MILHOJA_ERROR_INT_MAX_MISMATCH               103
#define MILHOJA_ERROR_BAD_FP32_SIZE                  104
#define MILHOJA_ERROR_BAD_FP32_EPSILON               105
#define MILHOJA_ERROR_BAD_FP32_DIGITS                106
#define MILHOJA_ERROR_BAD_FP32_EXPONENT              107
#define MILHOJA_ERROR_BAD_FP64_SIZE                  108
#define MILHOJA_ERROR_BAD_FP64_EPSILON               109
#define MILHOJA_ERROR_BAD_FP64_DIGITS                110
#define MILHOJA_ERROR_BAD_FP64_EXPONENT              111
#define MILHOJA_ERROR_REAL_SIZE_MISMATCH             112
#define MILHOJA_ERROR_REAL_EPSILON_MISMATCH          113
#define MILHOJA_ERROR_REAL_DIGITS_MISMATCH           114
#define MILHOJA_ERROR_REAL_EXPONENT_MISMATCH         115

#if 0
--------------------------------------------------------------------------------
Runtime-specific error codes
--------------------------------------------------------------------------------
#endif
#define MILHOJA_ERROR_UNABLE_TO_INIT_RUNTIME         200
#define MILHOJA_ERROR_UNABLE_TO_FINALIZE_RUNTIME     201
#define MILHOJA_ERROR_N_THREAD_TEAMS_NEGATIVE        202
#define MILHOJA_ERROR_N_THREADS_PER_TEAM_NEGATIVE    203
#define MILHOJA_ERROR_N_STREAMS_NEGATIVE             204
#define MILHOJA_ERROR_UNABLE_TO_EXECUTE_TASKS        205

#if 0
--------------------------------------------------------------------------------
Grid-specific error codes
--------------------------------------------------------------------------------
#endif
#define MILHOJA_ERROR_UNABLE_TO_INIT_GRID                  300
#define MILHOJA_ERROR_UNABLE_TO_FINALIZE_GRID              301
#define MILHOJA_ERROR_UNABLE_TO_INIT_DOMAIN                302
#define MILHOJA_ERROR_UNABLE_TO_GET_BOUNDS                 303
#define MILHOJA_ERROR_UNABLE_TO_GET_LEVEL                  304
#define MILHOJA_ERROR_UNABLE_TO_GET_DELTAS                 305
#define MILHOJA_ERROR_UNABLE_TO_WRITE_PLOTFILE             306
#define MILHOJA_ERROR_UNABLE_TO_GET_BLOCK_SIZE             307
#define MILHOJA_ERROR_UNABLE_TO_GET_N_GUARDCELLS           308
#define MILHOJA_ERROR_UNABLE_TO_GET_N_CC_VARS              309
#define MILHOJA_ERROR_UNABLE_TO_GET_DOMAIN_DECOMPOSITION   310
#define MILHOJA_ERROR_UNABLE_TO_GET_COORD_SYS              311
#define MILHOJA_ERROR_UNABLE_TO_FILL_GCS                   312
#define MILHOJA_ERROR_UNABLE_TO_GET_N_FLUX_VARS            313
#define MILHOJA_ERROR_INVALID_N_FLUX_VARS                  314

#if 0
--------------------------------------------------------------------------------
Tile-specific error codes
--------------------------------------------------------------------------------
#endif
#define MILHOJA_ERROR_UNABLE_TO_GET_METADATA         400

#if 0
--------------------------------------------------------------------------------
Iterator-specific error codes
--------------------------------------------------------------------------------
#endif
#define MILHOJA_ERROR_UNABLE_TO_BUILD_ITERATOR       500
#define MILHOJA_ERROR_UNABLE_TO_DESTROY_ITERATOR     501
#define MILHOJA_ERROR_UNABLE_TO_VALIDATE_ITERATOR    502
#define MILHOJA_ERROR_UNABLE_TO_ADVANCE_ITERATOR     503
#define MILHOJA_ERROR_UNABLE_TO_ACQUIRE_TILE         504
#define MILHOJA_ERROR_UNABLE_TO_RELEASE_TILE         505

