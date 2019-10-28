all: main.o bitmap.o
	mpicc -o main main.o bitmap.o

main.o: bitmap.o
	mpicc -c main.c

bitmap.o:
	mpicc -c bitmap.c

clean:
	rm -rf ./*.o
