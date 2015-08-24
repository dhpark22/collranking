#ifndef __ALTSVM_HPP__
#define __ALTSVM_HPP__

#include <random>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "../elements.hpp"
#include "../model.hpp"
#include "../ratings.hpp"
#include "../loss.hpp"
#include "../problem.hpp"
#include "../evaluator.hpp"
#include "solver.hpp"

class SolverAltSVM : public Solver {
  protected:
    double dcd_delta(loss_option_t, double, double, double, double);

  public:
    SolverAltSVM() : Solver() {}
    SolverAltSVM(init_option_t init, int n_th, int m_it = 0) : Solver(init, m_it, n_th) {}
    void solve(Problem&, Model&, Evaluator*);
};


double SolverAltSVM::dcd_delta(loss_option_t loss_option, double alpha, double a, double b, double C) {

  double delta;

  switch(loss_option) {
    case L1_HINGE:
      // closed-form solution
      delta = (1. - b) / a; 
      delta = min(max(0., alpha + delta), C) - alpha;
    case L2_HINGE:
      // closed-form solution
      delta = (1. - b - alpha*.5/C) / (a + .5/C);
      delta = max(0., alpha + delta) - alpha;      
      break;
    case LOGISTIC:
      // dual coordinate descent step
      delta = 0.;
      break;
    case SQUARED:
      // dual coordinate descent step
      delta = 0.;
  }

  return delta;

}

void SolverAltSVM::solve(Problem& prob, Model& model, Evaluator* eval) {

  double lambda = prob.lambda;
  
  n_users = prob.n_users;
  n_items = prob.n_items;
  n_train_comps = prob.n_train_comps; 

  int n_max_updates = n_train_comps/n_threads;

  double *alphaV = new double[this->n_train_comps];
  double *alphaU = new double[this->n_train_comps];
  memset(alphaU, 0, sizeof(double) * this->n_train_comps);
  memset(alphaV, 0, sizeof(double) * this->n_train_comps);

  // Alternating RankSVM
  double f, f_old;

  double time = omp_get_wtime();
  initialize(prob, model, init_option);
  time = omp_get_wtime() - time;

  printf("0, %f, ", time);
  f_old = prob.evaluate(model);
  eval->evaluate(model);
  printf("\n");

  double normsq;
  for (int OuterIter = 1; OuterIter <= max_iter; ++OuterIter) {

    ///////////////////////////
    // Learning V 
    ///////////////////////////
    
    double time_single_iter = omp_get_wtime(); 
    
    // initialize using the previous alphaV
    memset(model.V, 0, sizeof(double) * n_items * model.rank);
    
    #pragma omp parallel for
    for(int i=0; i<n_train_comps; ++i) {
      double *user_vec  = &(model.U[prob.train[i].user_id  * model.rank]);
      double *item1_vec = &(model.V[prob.train[i].item1_id * model.rank]);
      double *item2_vec = &(model.V[prob.train[i].item2_id * model.rank]);
      //if (alphaV[i] > 1e-10) {
        for(int j=0; j<model.rank; ++j) {
          double d = alphaV[i] * user_vec[j];
          item1_vec[j] += d;
          item2_vec[j] -= d;
        }
      //}
    }		

    // DUAL COORDINATE DESCENT for V
    #pragma omp parallel
    {
      int i_thread = omp_get_thread_num();

      std::mt19937 gen(n_threads*OuterIter + i_thread);
      std::uniform_int_distribution<int> randidx(0, n_train_comps-1);

      for(int n_updates=0; n_updates<n_max_updates; ++n_updates) {
        int idx = randidx(gen);
        double *user_vec  = &(model.U[prob.train[idx].user_id  * model.rank]);
        double *item1_vec = &(model.V[prob.train[idx].item1_id * model.rank]);
        double *item2_vec = &(model.V[prob.train[idx].item2_id * model.rank]);
    
        double p1 = 0., p2 = 0., d = 0.;
        for(int j=0; j<model.rank; ++j) {
          d = item1_vec[j] - item2_vec[j];
          p1 += user_vec[j] * d;
          p2 += user_vec[j] * user_vec[j];
        } 

        double delta = dcd_delta(prob.loss_option, alphaV[idx], p2*2., p1, 1./lambda);

        if (delta != 0.) { 
          alphaV[idx] += delta;
          for(int j=0; j<model.rank; ++j) {
            d = delta * user_vec[j];
            item1_vec[j] += d; 
            item2_vec[j] -= d;
          }
        }
      }

    }
    
    time = time + (omp_get_wtime() - time_single_iter);

    // compute performance measure
    printf("%d, %f, ", OuterIter, time);
    f = prob.evaluate(model);
    eval->evaluate(model);
    printf("\n");
 
    ///////////////////////////
    // Learning U 
    ///////////////////////////
    
    time_single_iter = omp_get_wtime();
 
    // initialize U using the previous alphaU 
    memset(model.U, 0, sizeof(double) * n_users * model.rank);
    
    #pragma omp parallel for
    for(int i=0; i<n_train_comps; ++i) {
      //if (alphaU[i] > 1e-10) {
        double *user_vec  = &(model.U[prob.train[i].user_id  * model.rank]);
        double *item1_vec = &(model.V[prob.train[i].item1_id * model.rank]);
        double *item2_vec = &(model.V[prob.train[i].item2_id * model.rank]);
        for(int j=0; j<model.rank; ++j) {
          user_vec[j] += alphaU[i] * (item1_vec[j] - item2_vec[j]);  
        }
      //}
    }

    // DUAL COORDINATE DESCENT for U
    #pragma omp parallel
    {
      int i_thread = omp_get_thread_num();
      int uid_from = (n_users * i_thread / n_threads);
      int uid_to   = (n_users * (i_thread+1) / n_threads);

      std::mt19937 gen(n_threads*OuterIter + i_thread);
      std::uniform_int_distribution<int> randidx(prob.tridx[uid_from], prob.tridx[uid_to]-1);

      for(int n_updates=0; n_updates<n_max_updates; ++n_updates) {
        int idx = randidx(gen);
        double *user_vec  = &(model.U[prob.train[idx].user_id  * model.rank]);
        double *item1_vec = &(model.V[prob.train[idx].item1_id * model.rank]);
        double *item2_vec = &(model.V[prob.train[idx].item2_id * model.rank]);
    
        double p1 = 0., p2 = 0., d = 0.;
        for(int j=0; j<model.rank; ++j) {
          d = item1_vec[j] - item2_vec[j];
          p1 += user_vec[j] * d;
          p2 += d*d;
        } 

        double delta = dcd_delta(prob.loss_option, alphaU[idx], p2, p1, 1./lambda);

        alphaU[idx] += delta;
        for(int j=0; j<model.rank; ++j) {
          d = delta * (item1_vec[j] - item2_vec[j]);
          user_vec[j] += d;
        }
      }
		}

    time = time + (omp_get_wtime() - time_single_iter);

    // compute performance measure 
    printf("%d, %f, ", OuterIter, time);
    f = prob.evaluate(model);
    eval->evaluate(model);
    printf("\n");
 
   // stopping rule
    if ((f_old - f) / f_old < 1e-5) break;
    f_old = f;
  
  }

	delete [] alphaV;
	delete [] alphaU;
}	

#endif
