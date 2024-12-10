# Defines Makefile variables that apply to all implementations

# set to use only smaller data sets
USE_ONLY_SMALLER_DATA_SETTING :=
#-DSMALLER_SETS_ONLY

# set to use limited set of test parameters and fewer runs for faster results for testing
USE_LIMITED_TEST_PARAMS_FEWER_RUNS := -DLIMITED_TEST_PARAMS_FEWER_RUNS

# define variable with all common implementation defines to use in
# implementation Makefiles
COMMON_IMP_DEFINES := ${USE_ONLY_SMALLER_DATA_SETTING} ${USE_LIMITED_TEST_PARAMS_FEWER_RUNS}

# define source code directory to use in Makefile
SRC_DIR = src

# define the path for the compiler
CC := g++

