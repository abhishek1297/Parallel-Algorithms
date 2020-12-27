################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CU_SRCS += \
../src/bfs/bfsGPU.cu \
../src/bfs/main.cu 

CPP_SRCS += \
../src/bfs/bfsCPU.cpp 

OBJS += \
./src/bfs/bfsCPU.o \
./src/bfs/bfsGPU.o \
./src/bfs/main.o 

CU_DEPS += \
./src/bfs/bfsGPU.d \
./src/bfs/main.d 

CPP_DEPS += \
./src/bfs/bfsCPU.d 


# Each subdirectory must supply rules for building sources it contributes
src/bfs/%.o: ../src/bfs/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/bin/nvcc -O3 -gencode arch=compute_61,code=sm_61  -odir "src/bfs" -M -o "$(@:%.o=%.d)" "$<"
	/usr/bin/nvcc -O3 --compile  -x c++ -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

src/bfs/%.o: ../src/bfs/%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/bin/nvcc -O3 -gencode arch=compute_61,code=sm_61  -odir "src/bfs" -M -o "$(@:%.o=%.d)" "$<"
	/usr/bin/nvcc -O3 --compile --relocatable-device-code=true -gencode arch=compute_61,code=compute_61 -gencode arch=compute_61,code=sm_61  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


