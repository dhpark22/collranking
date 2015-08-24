CC=g++
CFLAG=-std=c++11 -fopenmp -O3

col:
	$(CC) $(CFLAG) -o collrank code/collrank.cpp 

run:
	./collrank

clean:
	rm *.o[0-9]* *.e[0-9]* *.o collrank 
