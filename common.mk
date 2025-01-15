# Defines Makefile variables that apply to all implementations

# set to use only smaller data sets
USE_ONLY_SMALLER_DATA_SETTING :=
#-DSMALLER_SETS_ONLY

# set to only run fastest and not use alternate optimized implementations in evaluation
ALT_OPTIMIZED_IMP_SETTING =
#-DNO_ALT_OPTIMIZED_IMPS

# set to use only default parallel parameters or limited set of alternative
# parallel parameters for optimization
ALT_PARALLEL_PARAMS_SETTING :=
#-DLIMITED_ALT_PARALLEL_PARAMS
#-DDEFAULT_PARALLEL_PARAMS_ONLY

# set to use fewer runs per configuration for faster results (at expense of possible accuracy)
FEWER_RUNS_PER_CONFIG_SETTING :=
#-DFEWER_RUNS_PER_CONFIG

# set to use only run evaluation using specific datatype
# only can use one of these defines at a time
SINGLE_DATATYPE_ONLY_SETTING :=
#-DEVAL_FLOAT_DATATYPE_ONLY
#-DEVAL_DOUBLE_DATATYPE_ONLY
#-DEVAL_HALF_DATATYPE_ONLY

# set to only run evaluation using runs where the loop iteration counts are not templated
USE_NOT_TEMPLATED_LOOP_ITER_COUNTS_ONLY_SETTING :=
#-DEVAL_NOT_TEMPLATED_ITERS_ONLY

# define variable with all common implementation defines to use in
# implementation Makefiles
COMMON_IMP_DEFINES := ${USE_ONLY_SMALLER_DATA_SETTING} \
                      ${ALT_OPTIMIZED_IMP_SETTING} \
                      ${ALT_PARALLEL_PARAMS_SETTING} \
                      ${FEWER_RUNS_PER_CONFIG_SETTING} \
                      ${SINGLE_DATATYPE_ONLY_SETTING} \
                      ${USE_NOT_TEMPLATED_LOOP_ITER_COUNTS_ONLY_SETTING}

# define source code directory to use in Makefile
SRC_DIR = src

# define the path for the compiler
CC := g++
