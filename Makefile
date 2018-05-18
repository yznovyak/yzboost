CCFLAGS := -O3 -march=native -std=c++11 -fopenmp -fopenmp-simd

build:
	g++ $(CCFLAGS) src/*.cpp src/*.h -o yzboost
