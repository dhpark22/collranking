CC=g++
CFLAG=-std=c++11 -O3 -g -fopenmp

col:
	$(CC) $(CFLAG) collrank.cpp 
run:
	./a.out
clean:
	rm *.o[0-9]* *.e[0-9]* *.o a.out
