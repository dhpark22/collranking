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
#include "solver/global.hpp"

struct configuration {
  std::string algo = "alt_svm", loss = "l2hinge";
  std::string type_str = "numeric", train_comps_file, train_file, test_file;
  int rank = 10, n_threads = 1, max_iter = 10;
  double lambda = 1000, tol = 1e-5;
  double alpha, beta;
  bool evaluate_every_iter = true;
};

int readConf(struct configuration& conf, std::string conFile) {

  std::ifstream infile(conFile);
  std::string line;
  while(std::getline(infile,line))
  {
    std::istringstream iss(line);
    if ((line[0] == '#') || (line[0] == '[')) {
      continue;
    }
  
    std::string key, equal, val;
    if (iss >> key >> equal >> val) {
      if (equal != "=") {
        continue;
      }
      if (key == "type") {
        conf.type_str = val;
      }
      if (key == "train_pairwise_file") {
        conf.train_comps_file = val;
      }
      if (key == "train_file") {
        conf.train_file = val;
      }
      if (key == "test_file") {
        conf.test_file = val;
      }
      if (key == "algorithm") {
        conf.algo = val;
      }
      if (key == "loss") {
        conf.loss = val;
      }
      if (key == "lambda") {
        conf.lambda = std::stod(val);
      }
      if (key == "rank") {
        conf.rank = std::stoi(val);
      }
      if (key == "max_outer_iter") {
        conf.max_iter = std::stoi(val);
      }
      if (key == "tol") {
        conf.tol = std::stod(val);
      }
      if (key == "evaluate") {
        if (val == "true") conf.evaluate_every_iter = true;
        if (val == "false") conf.evaluate_every_iter = false;
      }
      if (key == "nthreads") {
        conf.n_threads = std::stoi(val);
      }
      if (key == "stepsize_alpha") {
        conf.alpha = std::stod(val);
      }
      if (key == "stepsize_beta") {
        conf.beta = std::stod(val);
      }
    }
  }

  return 1;
}

int main (int argc, char* argv[]) {
  struct configuration conf;
  std::string config_file = "config/default.cfg";

  if (argc > 2) {
    std::cerr << "Usage : " << std::string(argv[0]) << " [config_file]" << std::endl;
    return -1;
  }

  if (argc == 2) {
    config_file = std::string(argv[1]);
  }

  if (!readConf(conf, config_file)) {
    std::cerr << "Usage : " << std::string(argv[0]) << " [config_file]" << std::endl;
    return -1;
  }

  // Problem definition 
  Problem prob;
 
  if (conf.loss == "l1hinge")
    prob.loss_option = L1_HINGE;
  else if (conf.loss == "l2hinge")
    prob.loss_option = L2_HINGE;
  else if (conf.loss == "logistic")
    prob.loss_option = LOGISTIC;
  else if (conf.loss == "squared")
    prob.loss_option = SQUARED;
  else {
    std::cerr << "ERROR : provide correct loss function !\n";
    return 1;
  }
  
  if ((conf.type_str != "numeric") && (conf.type_str != "binary")) {
    cerr << "ERROR : provide correct experiment type !\n";
    return 1;
  }

  prob.lambda = conf.lambda;

  std::cout << "Loading training set file : " << conf.train_comps_file << std::endl;
  prob.read_data(conf.train_comps_file);

  // Model definition
  Model model(prob.get_nusers(), prob.get_nitems(), conf.rank);

  // Evaluator definition
  Evaluator* eval;

  vector<int> k_list;
  
  if (conf.type_str == "numeric") {
    eval = new EvaluatorRating;
    // current only ndcg@10 can be computed 
    k_list.push_back(10);
  }
  else if (conf.type_str == "binary") {
    eval = new EvaluatorBinary;
    k_list.push_back(1);
    k_list.push_back(5);
    k_list.push_back(10);
    k_list.push_back(100);
  } 

  std::cout << "Reading test set file : " << conf.test_file << std::endl;
  eval->load_files(conf.train_file, conf.test_file, k_list);

  // Solver definition
  omp_set_dynamic(0);
  omp_set_num_threads(conf.n_threads);

  Solver* mySolver;

  if (conf.algo == "altsvm") {
    printf("AltSVM with %d threads..\n", conf.n_threads);
    mySolver = new SolverAltSVM(INIT_RANDOM, conf.n_threads, conf.max_iter);
  }
  else if (conf.algo == "sgd") {
    printf("SGD with %d threads.. \n", conf.n_threads);
    mySolver = new SolverSGD(conf.alpha, conf.beta, INIT_RANDOM, conf.n_threads, conf.max_iter);
  }
  else if (conf.algo == "global") {
    printf("Global ranking with all-aggregated comparisons.. \n");
    mySolver = new SolverGlobal(INIT_RANDOM, conf.n_threads, conf.max_iter);
  }
  else {
    std::cerr << "ERROR : provide correct algorithm !\n";
    return -1;
  }

  if (conf.type_str == "numeric") {
    printf("iteration, training time (sec), pairwise error, ndcg@10\n");
  }
  else if (conf.type_str == "binary") {
    printf("iteration, training time (sec), precision@K\n"); 
  }

  mySolver->solve(prob, model, eval);
  delete mySolver;

  return 0;
}
