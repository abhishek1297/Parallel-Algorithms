################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../src/Astar/astarCPU.cpp 

OBJS += \
./src/Astar/astarCPU.o 

CPP_DEPS += \
./src/Astar/astarCPU.d 


# Each subdirectory must supply rules for building sources it contributes
src/Astar/%.o: ../src/Astar/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/bin/nvcc -Icuda -G -g -O0 -std=c++11 -gencode arch=compute_61,code=sm_61  -odir "src/Astar" -M -o "$(@:%.o=%.d)" "$<"
	/usr/bin/nvcc -Icuda -G -g -O0 -std=c++11 --compile  -x c++ -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


