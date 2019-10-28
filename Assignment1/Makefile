
all: clean main.o bitmap.o
	gcc -o bitmap main.o bitmap.o
	rm -rf ./*.o

main.o: bitmap.o
	gcc -c main.c

bitmap.o:
	gcc -c bitmap.c

clean:
	rm -rf ./*.o
