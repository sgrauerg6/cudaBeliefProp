# Defines Makefile variables that apply to all implementations

# set to use only smaller data sets
USE_ONLY_SMALLER_DATA_SETTING :=
#-DSMALLER_SETS_ONLY

# set to use limited set of test parallel parameters for optimization
LIMITED_TEST_PARAMS_SETTING :=
#-DLIMITED_TEST_PARAMS

# set to use fewer runs per configuration for faster results (at expense of possible accuracy)
FEWER_RUNS_PER_CONFIG_SETTING :=
#-DFEWER_RUNS_PER_CONFIG

# set to use only run evaluation using float datatype
FLOAT_DATATYPE_ONLY_SETTING :=
#-DEVAL_FLOAT_DATATYPE_ONLY

# set to only run evaluation using runs where the loop iteration counts are not templated
USE_NOT_TEMPLATED_LOOP_ITER_COUNTS_ONLY_SETTING :=
#-DEVAL_NOT_TEMPLATED_ITERS_ONLY

# define variable with all common implementation defines to use in
# implementation Makefiles
COMMON_IMP_DEFINES := ${USE_ONLY_SMALLER_DATA_SETTING} \
                      ${LIMITED_TEST_PARAMS_SETTING} \
                      ${FEWER_RUNS_PER_CONFIG_SETTING} \
					  ${FLOAT_DATATYPE_ONLY_SETTING} \
                      ${USE_NOT_TEMPLATED_LOOP_ITER_COUNTS_ONLY_SETTING}

# define source code directory to use in Makefile
SRC_DIR = src

# define the path for the compiler
CC := g++
