################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CU_SRCS += \
../src/main.cu 

CPP_SRCS += \
../src/dataIO.cpp \
../src/graph.cpp 

OBJS += \
./src/dataIO.o \
./src/graph.o \
./src/main.o 

CU_DEPS += \
./src/main.d 

CPP_DEPS += \
./src/dataIO.d \
./src/graph.d 


# Each subdirectory must supply rules for building sources it contributes
src/%.o: ../src/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/bin/nvcc -O3 -gencode arch=compute_61,code=sm_61  -odir "src" -M -o "$(@:%.o=%.d)" "$<"
	/usr/bin/nvcc -O3 --compile  -x c++ -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

src/%.o: ../src/%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/bin/nvcc -O3 -gencode arch=compute_61,code=sm_61  -odir "src" -M -o "$(@:%.o=%.d)" "$<"
	/usr/bin/nvcc -O3 --compile --relocatable-device-code=true -gencode arch=compute_61,code=compute_61 -gencode arch=compute_61,code=sm_61  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


