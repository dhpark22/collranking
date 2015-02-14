CC=g++
CFLAG=-std=c++11 -fopenmp
eigen=~/course/eigen/

col:
	$(CC) $(CFLAG) -g -O3 -I$(TACC_BOOST_INC) -L$(TACC_BOOST_LIB) -lboost_program_options -o collrank collrank.cpp 

run:
	./collrank

clean:
	rm *.o[0-9]* *.e[0-9]* *.o collrank 
