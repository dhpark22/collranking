CC=g++
CFLAG=-std=c++11 -O3 -g -fopenmp
eigen=~/course/eigen/

col:
	$(CC) $(CFLAG) -I$(TACC_BOOST_INC) -o collrank collrank.cpp 

run:
	./collrank

clean:
	rm *.o[0-9]* *.e[0-9]* *.o collrank 
