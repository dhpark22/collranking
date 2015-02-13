#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "problem.hpp"
#include "model.hpp"
#include "evaluator.hpp"
#include "solver/altsvm.hpp"
#include "solver/sgd.hpp"
#include <boost/program_options/option.hpp>

namespace po = boost::program_options;

int main (int argc, char* argv[]) {
  if (argc < 5) {
    cout << "Solve collaborative ranking problem with given training/testing data set" << endl;
    cout << "Usage ./collrank [training (comparison) file] [test (rating) file] [rank] [lambda] [num_threads]" << endl;
    return 0;
  }

  int rank = atoi(argv[3]);
  double lambda = atof(argv[4]);
  int n_threads = atoi(argv[5]);

  Problem prob(L2_HINGE, lambda);	

  prob.read_data(argv[1]);
  omp_set_dynamic(0);
  omp_set_num_threads(n_threads);
	
  double time;

  Model         model(prob.get_nusers(), prob.get_nitems(), rank);
  SolverAltSVM  altsvm_solver(INIT_RANDOM, n_threads, 50);
  SolverSGD     sgd_solver(1e-2, 1e-5, INIT_RANDOM, n_threads);

  EvaluatorRating eval;
  eval.load_files(argv[2]);

	altsvm_solver.solve(prob, model, eval);
/*
  sgd_solver.solve(prob, model, eval);

  prob.loss_option = LOGISTIC;
  sgd_solver.solve(prob, model, eval);
*/

  return 0;
}
