#ifndef __SGD_HPP__
#define __SGD_HPP__

#include <random>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <algorithm>
#include <vector>

#include "../elements.hpp"
#include "../model.hpp"
#include "../ratings.hpp"
#include "../loss.hpp"
#include "../problem.hpp"
#include "../evaluator.hpp"
#include "solver.hpp"

using namespace std;

class SolverSGD : public Solver {
  protected:
    double alpha, beta;
    
    vector<int> n_comps_by_user, n_comps_by_item;    

    bool sgd_step(Model&, const comparison&, loss_option_t, double, double);
 
  public:
    SolverSGD() : Solver() {}
    SolverSGD(double alp, double bet, init_option_t init, int n_th, int m_it = 0) : Solver(init, m_it, n_th), alpha(alp), beta(bet) {}
    void solve(Problem&, Model&, Evaluator* eval);
};

bool SolverSGD::sgd_step(Model& model, const comparison& comp, loss_option_t loss_option, double l, double step_size) {
  double *user_vec  = &(model.U[comp.user_id  * model.rank]);
  double *item1_vec = &(model.V[comp.item1_id * model.rank]);
  double *item2_vec = &(model.V[comp.item2_id * model.rank]);

  int n_comps_user  = n_comps_by_user[comp.user_id];
  int n_comps_item1 = n_comps_by_item[comp.item1_id];
  int n_comps_item2 = n_comps_by_item[comp.item2_id];

  if ((n_comps_user < 1) || (n_comps_item1 < 1) || (n_comps_item2 < 1)) printf("ERROR\n");

  double prod = 0.;
  for(int k=0; k<model.rank; k++) prod += user_vec[k] * comp.comp * (item1_vec[k] - item2_vec[k]);

  if (prod != prod) return false;

  double grad = 0.;
  switch(loss_option) {
    case L2_HINGE:
      grad = (prod<1.) ? 2.*(prod-1.):0.;
      break;
    case L1_HINGE:
      grad = (prod<1.) ? -1.:0.;
      break;
    case LOGISTIC:
      grad = -exp(-prod)/(1.+exp(-prod));
      break;
    case SQUARED:
      grad = 2.*(prod-1.);
  }

  if (grad != 0.) {
    for(int k=0; k<model.rank; k++) {
	    double user_dir  = step_size * (grad * comp.comp * (item1_vec[k] - item2_vec[k]) + l / (double)n_comps_user * user_vec[k]);
      double item1_dir = step_size * (grad * comp.comp * user_vec[k] + l / (double)n_comps_item1 * item1_vec[k]);
      double item2_dir = step_size * (grad * -comp.comp * user_vec[k] + l / (double)n_comps_item2 * item2_vec[k]);

	    user_vec[k]  -= user_dir;
	    item1_vec[k] -= item1_dir;
      item2_vec[k] -= item2_dir;
    }
  }

  return true;
}

void SolverSGD::solve(Problem& prob, Model& model, Evaluator* eval) { 

  n_users = prob.n_users;
  n_items = prob.n_items;
  n_train_comps = prob.n_train_comps;

  n_comps_by_user.resize(n_users,0);
  n_comps_by_item.resize(n_items,0);
  for(int i=0; i<n_train_comps; ++i) {
    ++n_comps_by_user[prob.train[i].user_id];
    ++n_comps_by_item[prob.train[i].item1_id];
    ++n_comps_by_item[prob.train[i].item2_id];
  } 
 
  double time = omp_get_wtime();
  initialize(prob, model, init_option); 
  time = omp_get_wtime() - time;

  double f;
  printf("0, %f, ", time);
  f = prob.evaluate(model);
  eval->evaluate(model);
  printf("\n");

  int n_max_updates = n_train_comps/n_threads;

  bool flag = false;

  for(int iter=0; iter<max_iter; ++iter) {
    double time_single_iter = omp_get_wtime();
    #pragma omp parallel
    {
      std::mt19937 gen(n_threads*iter+omp_get_thread_num());
      std::uniform_int_distribution<int> randidx(0, n_train_comps-1);

      for(int n_updates=1; n_updates<n_max_updates; ++n_updates) {
        int idx = randidx(gen);
        double stepsize = alpha/(1.+beta*(double)((n_updates+n_max_updates*iter)*n_threads));
        if (!sgd_step(model, prob.train[idx], prob.loss_option, prob.lambda, stepsize)) {
          flag = true;
          break;
        }
      }
    }

    if (flag) break;

    time = time + (omp_get_wtime() - time_single_iter);
    printf("%d, %f, ", iter+1, time);
    f = prob.evaluate(model);
    eval->evaluate(model);
    printf("\n");
    
  } 

}



#endif
