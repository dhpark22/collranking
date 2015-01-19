// Project: Parallel Collaborative Ranking with AltSVM and SGD
// Collaborative work by Dohyung Park and Jin Zhang
// Date: 11/26/2014
//
// The script will:
// [1a] solve the problem with alternative rankSVM via liblineaer
// [1b] solve the problem with stochasitic gradient descent in hogwild style
// [1c] solve the problem with stochastic gradient descent in nomad style
//
// Run: ./a.out [rating_file] [rating_format] [num_partitions]

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "problem.hpp"

int main (int argc, char* argv[]) {
	if (argc < 5) {
		cout << "Solve collaborative ranking problem with given training/testing data set" << endl;
		cout << "Usage ./collaborative_ranking [rating file] [training file] [testing file] [num_threads]" << endl;
		return 0;
	}

  int rank = atoi(argv[3]);
	int n_threads = atoi(argv[4]);

	Problem p(rank, n_threads);	

  printf("Reading data files..\n");

	p.read_data(argv[1], argv[2]);
	omp_set_dynamic(0);
	omp_set_num_threads(n_threads);
	
  double time;
 
  time = omp_get_wtime(); 
  printf("Running AltSVM with random init.. \n");  
	p.run_altsvm(1000., INIT_RANDOM);

  time = omp_get_wtime(); 
  printf("Running AltSVM with all ones init.. \n");  
	p.run_altsvm(1000., INIT_ALLONES);

  time = omp_get_wtime(); 
  printf("Running AltSVM with svd init.. \n");  
	p.run_altsvm(1000., INIT_SVD);
/*
  time = omp_get_wtime();
  printf("Running Random SGD with random init.. \n");
  p.run_sgd_random(1000., 1e-1, 1e-5, INIT_RANDOM);

  time = omp_get_wtime();
  printf("Running Random SGD with SVD init.. \n");
  p.run_sgd_random(1000., 1e-1, 1e-5, INIT_SVD);

  time = omp_get_wtime();
  printf("Running Random SGD with SVD init.. \n");
  p.run_sgd_random(1000., 1e-2, 1e-5, INIT_SVD);

  time = omp_get_wtime();
  printf("Running NOMADi SGD.. \n");
  p.run_sgd_nomad_item(100., 1e-1, 1e-5, INIT_RANDOM);
  
  time = omp_get_wtime();
  printf("Running NOMADu SGD.. \n");
  p.run_sgd_nomad_user(100., 1e-1, 1e-5, INIT_RANDOM);
*/

  return 0;
}
