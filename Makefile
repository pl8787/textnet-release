CONFIG_FILE := Makefile.config
include $(CONFIG_FILE)

# set LD_LIBRARY_PATH
export CC  = gcc
export CXX = g++
export NVCC = $(CUDA_DIR)/bin/nvcc

# Custom compiler
ifdef CUSTOM_CXX
    CXX := $(CUSTOM_CXX)
endif

CUDA_INCLUDE_DIR := $(CUDA_DIR)/include
CUDA_LIB_DIR := $(CUDA_DIR)/lib64

INCLUDE_DIRS += $(CUDA_INCLUDE_DIR)
LIBRARY_DIRS += $(CUDA_LIB_DIR)

# Complete build flags.
COMMON_FLAGS += $(foreach includedir,$(INCLUDE_DIRS),-I$(includedir))
CXXFLAGS += -pthread -fPIC $(COMMON_FLAGS) $(WARNINGS)
NVCCFLAGS += -ccbin=$(CXX) -Xcompiler -fPIC $(COMMON_FLAGS) --use_fast_math -g -O3
LINKFLAGS += -fPIC $(COMMON_FLAGS) $(WARNINGS)
LDFLAGS += $(foreach librarydir,$(LIBRARY_DIRS),-L$(librarydir)) \
        $(foreach library,$(LIBRARIES),-l$(library))

CXXFLAGS += -Wall -g -O3 -msse3 -Wno-unknown-pragmas -funroll-loops -I../mshadow/

LDFLAGS += -lm -lcudart -lcublas -lmkl_core -lmkl_intel_lp64 -lmkl_intel_thread -liomp5 -lpthread -lcurand -lz `pkg-config --libs opencv`

export NVCCFLAGS = --use_fast_math -g -O3 -ccbin $(CXX)


# specify tensor path
BIN = bin/textnet
OBJ = layer_cpu.o initializer_cpu.o updater_cpu.o
#  nnet_cpu.o data.o
CUOBJ = layer_gpu.o initializer_gpu.o updater_gpu.o
#  nnet_gpu.o

all: $(BIN)

layer_cpu.o layer_gpu.o: src/layer/layer_impl.cpp src/layer/layer_impl.cu\
	src/layer/*.h src/layer/*.hpp src/layer/common/*.hpp src/utils/*.h

updater_cpu.o updater_gpu.o: src/updater/updater_impl.cpp src/updater/updater_impl.cu\
	src/updater/*.hpp src/updater/*.h src/utils/*.h
  
initializer_cpu.o initializer_gpu.o: src/initializer/initializer_impl.cpp src/initializer/initializer_impl.cu\
  src/initializer/*.hpp src/initializer/*.h src/utils/*.h

# nnet_cpu.o nnet_gpu.o: src/nnet/nnet_impl.cpp src/nnet/nnet_impl.cu src/layer/layer.h\
	# src/updater/updater.h src/utils/*.h src/nnet/*.hpp src/nnet/*.h

# data.o: src/io/data.cpp src/io/*.hpp

bin/textnet: src/textnet_main.cpp $(OBJ) $(CUOBJ)

$(BIN) :
	$(CXX) $(CXXFLAGS)  -o $@ $(filter %.cpp %.o %.c %.a, $^) $(LDFLAGS)

$(OBJ) :
	$(CXX) -c $(CXXFLAGS) -o $@ $(firstword $(filter %.cpp %.c, $^) )

$(CUOBJ) :
	$(NVCC) -c -o $@ $(NVCCFLAGS) -Xcompiler "$(CXXFLAGS)" $(filter %.cu, $^)

$(CUBIN) :
	$(NVCC) -o $@ $(NVCCFLAGS) -Xcompiler "$(CXXFLAGS)" -Xlinker "$(LDFLAGS)" $(filter %.cu %.cpp %.o, $^)
  
clean:
	$(RM) $(OBJ) $(BIN) $(CUBIN) $(CUOBJ) *~ */*~ */*/*~



