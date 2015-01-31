// Project: Parallel Collaborative Ranking with AltSVM and SGD
// Collaborative work by Dohyung Park and Jin Zhang
// Date: 11/26/2014
//
// The script will:
// [1a] solve the problem with alternative rankSVM via liblineaer
// [1b] solve the problem with stochasitic gradient descent in hogwild style
// [1c] solve the problem with stochastic gradient descent in nomad style
//
// Run: ./collrank [training (comparison) file] [test (rating) file] [rank] [num_threads]
	
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "problem.hpp"
#include "pk.hpp"

int main (int argc, char* argv[]) {
  if (argc < 5) {
    cout << "Solve collaborative ranking problem with given training/testing data set" << endl;
    cout << "Usage ./collrank [training (comparison) file] [test (rating) file] [rank] [lambda] [num_threads]" << endl;
    return 0;
  }

  int rank = atoi(argv[3]);
  double lambda = atof(argv[4]);
  int n_threads = atoi(argv[5]);

  Problem p(rank, n_threads);	

  printf("Reading data files..\n");

  p.read_data(argv[1]);
  omp_set_dynamic(0);
  omp_set_num_threads(n_threads);
	
  double time;

  EvaluatorRating eval;
  eval.load_files(argv[2]);

  time = omp_get_wtime(); 
  printf("Running AltSVM with random init.. \n");  
	p.run_altsvm(eval, L2_HINGE, lambda, INIT_RANDOM);

  time = omp_get_wtime(); 
  printf("Running AltSVM with svd init.. \n");  
	p.run_altsvm(eval, L2_HINGE, lambda, INIT_SVD);
/*
  time = omp_get_wtime(); 
  printf("Running AltSVM with all-ones init.. \n");  
	p.run_altsvm(eval, L2_HINGE, lambda, INIT_ALLONES);

  time = omp_get_wtime();
  printf("Running Random SGD with SVD init.. \n");
  p.run_sgd_random(L2_HINGE, lambda, 1e-1, 1e-5, INIT_SVD);

  time = omp_get_wtime();
  printf("Running NOMADi SGD.. \n");
  p.run_sgd_nomad_item(100., 1e-1, 1e-5, INIT_RANDOM);
  
  time = omp_get_wtime();
  printf("Running NOMADu SGD.. \n");
  p.run_sgd_nomad_user(100., 1e-1, 1e-5, INIT_RANDOM);
*/

  return 0;
}
