################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CU_SRCS += \
../src/sssp/ssspCPU.cu \
../src/sssp/ssspGPU.cu 

CPP_SRCS += \
../src/sssp/ssspMain.cpp 

OBJS += \
./src/sssp/ssspCPU.o \
./src/sssp/ssspGPU.o \
./src/sssp/ssspMain.o 

CU_DEPS += \
./src/sssp/ssspCPU.d \
./src/sssp/ssspGPU.d 

CPP_DEPS += \
./src/sssp/ssspMain.d 


# Each subdirectory must supply rules for building sources it contributes
src/sssp/%.o: ../src/sssp/%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/bin/nvcc -Icuda -G -g -O0 -std=c++11 -gencode arch=compute_61,code=sm_61  -odir "src/sssp" -M -o "$(@:%.o=%.d)" "$<"
	/usr/bin/nvcc -Icuda -G -g -O0 -std=c++11 --compile --relocatable-device-code=true -gencode arch=compute_61,code=compute_61 -gencode arch=compute_61,code=sm_61  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

src/sssp/%.o: ../src/sssp/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/bin/nvcc -Icuda -G -g -O0 -std=c++11 -gencode arch=compute_61,code=sm_61  -odir "src/sssp" -M -o "$(@:%.o=%.d)" "$<"
	/usr/bin/nvcc -Icuda -G -g -O0 -std=c++11 --compile  -x c++ -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


