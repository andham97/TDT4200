PATH+=:/usr/local/cuda/bin
all: mandel
mandel:
	nvcc -o mandel mandel.cu -O3 -lm
clean:
	-rm -f mandel_c mandel

