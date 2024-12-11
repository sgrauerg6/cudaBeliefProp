# Defines Makefile variables that apply to all implementations

# set to use only smaller data sets
USE_ONLY_SMALLER_DATA_SETTING := -DSMALLER_SETS_ONLY

# set to use limited set of test parameters and fewer runs for faster results for testing
USE_LIMITED_TEST_PARAMS_FEWER_RUNS_SETTING := -DLIMITED_TEST_PARAMS_FEWER_RUNS

# set to use only run evaluation using float datatype
USE_FLOAT_DATATYPE_ONLY_SETTING :=
#-DEVAL_FLOAT_DATATYPE_ONLY

# set to only run evaluation using runs where the loop iteration counts are not templated
USE_NOT_TEMPLATED_LOOP_ITER_COUNTS_ONLY_SETTING :=
#-DEVAL_NOT_TEMPLATED_ITERS_ONLY

# define variable with all common implementation defines to use in
# implementation Makefiles
COMMON_IMP_DEFINES := ${USE_ONLY_SMALLER_DATA_SETTING} \
                      ${USE_LIMITED_TEST_PARAMS_FEWER_RUNS_SETTING} \
					  ${USE_FLOAT_DATATYPE_ONLY_SETTING} \
                      ${USE_NOT_TEMPLATED_LOOP_ITER_COUNTS_ONLY_SETTING}

# define source code directory to use in Makefile
SRC_DIR = src

# define the path for the compiler
CC := g++
