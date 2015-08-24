#ifndef __SOLVER_HPP__
#define __SOLVER_HPP__

#include <stdlib.h>
#include "../problem.hpp"
#include "../model.hpp"
#include "../evaluator.hpp"

enum init_option_t {INIT_RANDOM, INIT_SVD, INIT_ALLONES};

class Solver {

protected:
  int             n_users, n_items, n_train_comps;

  init_option_t   init_option;
  int             max_iter;

  int             n_threads;

  void initialize(Problem&, Model&, init_option_t);

public:
  Solver() {}
  Solver(init_option_t init, int m_it, int n_th) : n_users(0), n_items(0), n_train_comps(0), 
                                                   init_option(init), max_iter(m_it), n_threads(n_th) {}
  virtual void solve(Problem&, Model&, Evaluator* eval) = 0; 

};

void Solver::initialize(Problem& prob, Model& model, init_option_t option) {

  switch(option) {
    case INIT_ALLONES:

    for(int i=0; i<n_users*model.rank; i++) model.U[i] = 1./sqrt((double)model.rank);
    memset(model.V, 0, sizeof(double) * n_items * model.rank);
    break;

    case INIT_RANDOM:
  
    srand(time(NULL)); 
    for(int i=0; i<n_users*model.rank; i++) model.U[i] = (double)rand() / (double)RAND_MAX / sqrt((double)model.rank);
    for(int i=0; i<n_items*model.rank; i++) model.V[i] = (double)rand() / (double)RAND_MAX / sqrt((double)model.rank);
    break;
 
    case INIT_SVD:
    /* NOT IMPLEMENTED */
    srand(time(NULL)); 
    for(int i=0; i<n_users*model.rank; i++) model.U[i] = (double)rand() / (double)RAND_MAX / sqrt((double)model.rank);
    for(int i=0; i<n_items*model.rank; i++) model.V[i] = (double)rand() / (double)RAND_MAX / sqrt((double)model.rank);

  }

}

#endif

