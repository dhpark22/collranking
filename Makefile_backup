CC=g++
CFLAG=-std=c++11 -O3 -g -fopenmp
metis=$(WORK)/opt/metis
parmetis=$(WORK)/opt/parmetis
liblinear=$(WORK)/opt/liblinear-1.94

col:
	$(CC) $(CFLAG) -I$(liblinear) -c collaborative_ranking.cpp 
	$(CC) $(CFLAG) -o a.out collaborative_ranking.o $(liblinear)/linear.o $(liblinear)/tron.o $(liblinear)/blas/*.o
all:
	$(CC) $(CFLAG) -I$(liblinear) -c collaborative_ranking_seq.cpp 
	$(CC) $(CFLAG) -o seqcol collaborative_ranking_seq.o $(liblinear)/linear.o $(liblinear)/tron.o $(liblinear)/blas/*.o
	$(CC) $(CFLAG) -I$(liblinear) -c collaborative_ranking_seq2.cpp 
	$(CC) $(CFLAG) -o seqcol2 collaborative_ranking_seq2.o $(liblinear)/linear.o $(liblinear)/tron.o $(liblinear)/blas/*.o
	$(CC) $(CFLAG) -I$(liblinear) -c collaborative_ranking.cpp 
	$(CC) $(CFLAG) -o a.out collaborative_ranking.o $(liblinear)/linear.o $(liblinear)/tron.o $(liblinear)/blas/*.o
seq:
	$(CC) $(CFLAG) -I$(liblinear) -c collaborative_ranking_seq.cpp 
	$(CC) $(CFLAG) -o seqcol collaborative_ranking_seq.o $(liblinear)/linear.o $(liblinear)/tron.o $(liblinear)/blas/*.o
seq2:
	$(CC) $(CFLAG) -I$(liblinear) -c collaborative_ranking_seq2.cpp 
	$(CC) $(CFLAG) -o seqcol2 collaborative_ranking_seq2.o $(liblinear)/linear.o $(liblinear)/tron.o $(liblinear)/blas/*.o
run:
	./a.out
clean:
	rm *.o[0-9]* *.e[0-9]* *.o a.out
