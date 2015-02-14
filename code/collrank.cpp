#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <iterator>
#include <string>
#include "problem.hpp"
#include "model.hpp"
#include "evaluator.hpp"
#include "solver/altsvm.hpp"
#include "solver/sgd.hpp"
#include <boost/program_options.hpp>
namespace po = boost::program_options;

int main (int argc, char* argv[]) {

  int rank, n_threads, max_iter;
  double lambda, tol;
  std::string algo, loss;
  std::string train_file, test_file;
  double alpha, beta;

  try {
    std::string config_file;

    po::options_description generic("Generic options");
    generic.add_options()
      ("help", "produce help message")
      ("config,c", po::value<std::string>(&config_file)->default_value("../config/default.cfg"), "configuration file")
    ;

    po::options_description config("Configuration");
    config.add_options()
      ("input.train_file,f", po::value<std::string>(&train_file), "training set filename")
      ("input.test_file,t", po::value<std::string>(&test_file), "test set filename")
    ;

    po::options_description hidden("Config file options");
    hidden.add_options()
      ("prob.lambda", po::value<double>(&lambda)->default_value(1000), "regularization parameter")
      ("prob.rank", po::value<int>(&rank)->default_value(10), "rank of the matrix model")
      ("prob.algorithm", po::value<std::string>(&algo)->default_value("altsvm"), "algorithm")
      ("prob.loss", po::value<std::string>(&loss)->default_value("l2hinge"), "loss function")
      ("prob.max_outer_iter", po::value<int>(&max_iter)->default_value(10), "maximum number of outer iterations")
      ("prob.tol", po::value<double>(&tol)->default_value(1e-5), "tolerance for stopping")
      ("prob.evaluate", po::value<int>(), "whether evaluated at each outer iteration")
      ("par.nthreads", po::value<int>(&n_threads)->default_value(1), "number of openmp threads")
      ("sgd.stepsize_alpha", po::value<double>(&alpha), "SGD step size parameter alpha")
      ("sgd.stepsize_beta", po::value<double>(&beta), "SGD step size parameter beta")
    ;


    po::options_description cmdline_options;
    cmdline_options.add(generic).add(config).add(hidden);

    po::options_description cfgfile_options;
    cfgfile_options.add(config).add(hidden);

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, cmdline_options), vm);
    po::notify(vm);

    ifstream ifs(config_file.c_str());
    if (!ifs)
    {
      std::cout << "ERROR : cannot open config file: " << config_file << "\n";
      return 0;
    }
    else
    {
      store(parse_config_file(ifs, cfgfile_options), vm);
      notify(vm);
    }
    
    if (vm.count("help")) {
      std::cout << config << "\n";
      return 0;
    }

  }
  catch(exception& e) {
    cerr << "ERROR : " << e.what() << "\n";
    return 1;
  }
  catch(...) {
    cerr << "Exception of unknown type!\n";
  }

  // Problem definition 
  Problem prob;
 
  if (loss == "l1hinge")
    prob.loss_option = L1_HINGE;
  else if (loss == "l2hinge")
    prob.loss_option = L2_HINGE;
  else if (loss == "logistic")
    prob.loss_option = LOGISTIC;
  else if (loss == "squared")
    prob.loss_option = SQUARED;
  else {
    cerr << "ERROR : provide correct loss function !\n";
    return 1;
  }

  prob.lambda = lambda;

  std::cout << "Loading training set file : " << train_file << std::endl;
  prob.read_data(train_file);

  // Model definition
  Model model(prob.get_nusers(), prob.get_nitems(), rank);

  // Evaluator definition
  EvaluatorRating eval;
  
  std::cout << "Reading test set file : " << test_file << std::endl;
  eval.load_files(test_file);

  // Solver definition
  omp_set_dynamic(0);
  omp_set_num_threads(n_threads);

  if (algo == "altsvm") {
    SolverAltSVM solver(INIT_RANDOM, n_threads, max_iter);
    solver.solve(prob, model, eval);
  }
  else if (algo == "sgd") {
    SolverSGD solver(alpha, beta, INIT_RANDOM, n_threads, max_iter);
    solver.solve(prob, model, eval);
  }
  else if (algo == "global") {
//    SolverGlobal solver(INIT_RANDOM, n_threads, max_iter);
//    solver.solve(prob, model, eval);
  }
  else {
    cerr << "ERROR : provide correct algorithm !\n";
    return 1;
  }

  return 0;
}
