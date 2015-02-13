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
#include "model.hpp"
#include "evaluator.hpp"
#include "solver/altsvm.hpp"

int main (int argc, char* argv[]) {
  if (argc < 7) {
    cout << "Solve collaborative ranking problem with given training/testing data set" << endl;
    cout << "Usage ./collrank [training (comparison) file] [training (pairwise) file] [testing (pairwise) file] [rank] [lambda] [num_threads]" << endl;
    return 0;
  }

  int rank = atoi(argv[4]);
  double lambda = atof(argv[5]);
  int n_threads = atoi(argv[6]);

  Problem prob(L2_HINGE, lambda);	

  printf("Reading data files..\n");

  prob.read_data(argv[1]);
  omp_set_dynamic(0);
  omp_set_num_threads(n_threads);
	
  double time;

  Model model(prob.get_nusers(), prob.get_nitems(), rank);

  EvaluatorBinary eval;
  int vints[] = {1, 2, 3, 4, 5, 10, 100, 200, 500};
  vector<int> v(vints, vints + sizeof(vints) / sizeof(int) );

  double s1 = omp_get_wtime();
  eval.load_files(argv[2], argv[3], v);
  double ltime = omp_get_wtime() - s1;
  printf("loading time is %f\n\n", ltime);


  SolverAltSVM solver1(INIT_RANDOM, n_threads, 30);

  time = omp_get_wtime(); 
  printf("Running AltSVM with random init.. \n");  
  solver1.solve(prob, model, eval);
  //time = omp_get_wtime() - time;
  //printf("overall time %f\n", time);

  //-- time = omp_get_wtime(); 
  //-- printf("Running AltSVM with svd init.. \n");  
	//-- p.run_altsvm(eval, L2_HINGE, lambda, INIT_SVD);

  //-- time = omp_get_wtime(); 
  //-- printf("Running AltSVM with all-ones init.. \n");  
	//-- p.run_altsvm(eval, L2_HINGE, lambda, INIT_ALLONES);

  //time = omp_get_wtime();
  //printf("Running Random SGD with SVD init.. \n");
  //p.run_sgd_random(eval, L2_HINGE, lambda, atoi(argv[4]), atoi(argv[5]), INIT_RANDOM);
  //
  //
  //printf("Running BPR with random init.. \n");  
	//p.run_bpr(eval, LOGISTIC, lambda, INIT_RANDOM);

/*
  time = omp_get_wtime();
  printf("Running NOMADi SGD.. \n");
  p.run_sgd_nomad_item(100., 1e-1, 1e-5, INIT_RANDOM);
  
  time = omp_get_wtime();
  printf("Running NOMADu SGD.. \n");
  p.run_sgd_nomad_user(100., 1e-1, 1e-5, INIT_RANDOM);
*/

  return 0;
}