#ifndef __PROBLEM_HPP__
#define __PROBLEM_HPP__

#include <queue>
#include <random>
#include <functional>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <algorithm>
#include <map>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>

#include "elements.hpp"
#include "model.hpp"
#include "ratings.hpp"
#include "loss.hpp"
#include "pk.hpp"

using namespace std;

enum init_option_t {INIT_RANDOM, INIT_SVD, INIT_ALLONES};

class Problem {
  protected:
    int n_users, n_items, n_train_comps; 	          // number of users/items in training sample, number of samples in traing and testing data set
    int n_threads;                                  // parameters
    double lambda;

    Model model;
    loss_option_t loss_option = L2_HINGE;

    vector<comparison>   train;
    vector<int>          tridx;

    RatingMatrix tr;

    double dcd_delta(loss_option_t, double, double, double, double);
    void de_allocate();					            // deallocate U, V when they are used multiple times by different methods
    void initialize(init_option_t); 
    
    comparison random_comp(int, int);
    std::mt19937 random_comp_gen;
 
  public:
    Problem(int, int);				// default constructor
    ~Problem();					// default destructor
    void read_data(char*);	// read function
    void run_global(Evaluator&, loss_option_t, double, int); 
    void run_altsvm(Evaluator&, loss_option_t, double, init_option_t, int);
    double compute_objective();
};

// may be more parameters can be specified here
Problem::Problem (int r, int np) : model(r) { 
  n_threads = np;
  random_comp_gen.seed(time(NULL));
}

Problem::~Problem () {
  model.de_allocate();
}

void Problem::read_data(char *train_file) {

  // Prepare to read files
  n_users = n_items = 0;
  ifstream f;

  // Read training comparisons
  f.open(train_file);
  if (f.is_open()) {
    int uid, i1id, i2id, uid_current = 0;
    tridx.resize(0);
    tridx.push_back(0);
    while (f >> uid >> i1id >> i2id) {
      n_users = max(uid, n_users);
      n_items = max(i1id, max(i2id, n_items));
      --uid; --i1id; --i2id; // now user_id and item_id starts from 0

      while(uid > uid_current) {
        std::sort(train.begin()+tridx[uid_current], train.end(), comp_userwise);
        tridx.push_back(train.size());
        ++uid_current;
      }

      train.push_back(comparison(uid, i1id, i2id, 1));
    }

    std::sort(train.begin()+tridx[uid_current], train.end(), comp_userwise);
    tridx.push_back(train.size());
   
    n_train_comps = train.size();

  } else {
    printf("Error in opening the training file!\n");
    exit(EXIT_FAILURE);
  }
  f.close();

  printf("Read %d training comparisons\n", n_train_comps);
  printf("%d users, %d items\n", n_users, n_items);

/*
  // Construct tridx
  tridx.resize(n_users+1);
  sort(train.begin(), train.end(), comp_userwise);
  tridx[0] = 0;
  tridx[n_users] = n_train_comps;
  for(int idx=1; idx<n_train_comps; ++idx) if (train[idx-1].user_id < train[idx].user_id) tridx[train[idx].user_id] = idx;
*/
  // memory allocation
  model.allocate(n_users, n_items);
  printf("read file done\n");

}	

double Problem::compute_objective() {
  return compute_loss(model, train, loss_option) + .5 * lambda * (model.Unormsq() + model.Vnormsq());		
}

double Problem::dcd_delta(loss_option_t loss_option, double alpha, double a, double b, double C) {

  double delta;

  switch(loss_option) {
    case L1_HINGE:
      // closed-form solution
      delta = -(b - 1.) / a; 
      delta = std::min(std::max(0., alpha + delta), C) - alpha;
      break;
    case L2_HINGE:
      // closed-form solution
      delta = -(b + alpha*.5/C - 1.) / (a + .5/C);
      delta = std::max(0., alpha + delta) - alpha;    
      break;
    case LOGISTIC:
      // dual coordinate descent step
      delta = 0.;
      //printf("%f ", alpha);

      for(int i=0; i<3; ++i) {
        double f = (alpha+delta)*log(alpha+delta) + (C-alpha-delta)*log(C-alpha-delta) + a/2*delta*delta + b*delta; 
        double d = -(b + a*delta + log(alpha+delta) / log(C-alpha-delta)) / (a + C / (alpha+delta) / (C-alpha-delta));

        //printf("%f,%f", d, -(b + a*delta + log(alpha+delta) / log(C-alpha-delta)));
  
        d = std::min(std::max(1e-10, alpha + delta + d), C - 1e-10) - alpha - delta;
        while(f < (alpha+delta+d)*log(alpha+delta+d) + (C-alpha-delta-d)*log(C-alpha-delta-d) + a/2*(delta+d)*(delta+d) + b*(delta+d)) 
          d /= 2.;

        delta += d;

        //printf("%f/%f ", alpha+delta, (alpha+delta)*log(alpha+delta) + (C-alpha-delta)*log(C-alpha-delta) + a/2*delta*delta + b*delta - alpha*log(alpha) - (C-alpha)*log(C-alpha));
      }
      //printf("\n");

      break;
    case SQUARED:
      // dual coordinate descent step
      delta = 0.;
  }

  return delta;

}

void Problem::run_global(Evaluator& eval, loss_option_t loss_option = L2_HINGE, double l = 10., int MaxIter = 10) {

  printf("Global rank aggregation with %d threads.. \n", n_threads);

  lambda = l;

  int n_max_updates = n_train_comps/n_threads;

  double *alphaV = new double[this->n_train_comps];
  memset(alphaV, 0, sizeof(double) * this->n_train_comps);

  // Alternating RankSVM
  double start = omp_get_wtime();
  double f, f_old;

  initialize(INIT_ALLONES);
  memset(model.V, 0, sizeof(double) * n_items * model.rank);
  
  f_old = compute_loss(model, train, loss_option) + .5 * lambda * (model.Unormsq() + model.Vnormsq());
  printf("0, %f / %f, %f, %f, %f", omp_get_wtime() - start, f_old, model.Unormsq(), model.Vnormsq(), compute_loss(model, train, loss_option));
  eval.evaluate(model);
  printf("\n");

  double normsq;
  for (int OuterIter = 1; OuterIter <= MaxIter; ++OuterIter) {

    ///////////////////////////
    // Learning V 
    ///////////////////////////
     
    // DUAL COORDINATE DESCENT for V
    #pragma omp parallel
    {
      int i_thread = omp_get_thread_num();

      std::mt19937 gen(n_threads*OuterIter + i_thread);
      std::uniform_int_distribution<int> randidx(0, n_train_comps-1);

      for(int n_updates=0; n_updates<n_max_updates; ++n_updates) {
        int idx = randidx(gen);
        double *user_vec  = &(model.U[train[idx].user_id  * model.rank]);
        double *item1_vec = &(model.V[train[idx].item1_id * model.rank]);
        double *item2_vec = &(model.V[train[idx].item2_id * model.rank]);
    
        double p1 = 0., p2 = 0., d = 0.;
        for(int j=0; j<model.rank; ++j) {
          d = item1_vec[j] - item2_vec[j];
          p1 += user_vec[j] * d;
          p2 += user_vec[j] * user_vec[j];
        } 

        double delta = dcd_delta(loss_option, alphaV[idx], p2*2., p1, 1./lambda);

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

    // compute performance measure
    f = compute_loss(model, train, loss_option) + .5*lambda*(model.Unormsq() + model.Vnormsq());
    printf("%d, %f / %f, %f, %f, %f", OuterIter, omp_get_wtime() - start, f, model.Unormsq(), model.Vnormsq(), compute_loss(model, train, loss_option));
    eval.evaluate(model);
    printf("\n");
 
    // stopping rule
    if ((f_old - f) / f_old < 1e-5) break;
    f_old = f;
  
  }

	delete [] alphaV;
}	



void Problem::run_altsvm(Evaluator& eval, loss_option_t loss_option, double l = 10., init_option_t init_option = INIT_RANDOM, int MaxIter = 30) {

  printf("Alternating rankSVM with %d threads.. \n", n_threads);
  printf("lambda : %f \n", l);

  lambda = l;

  int n_max_updates = n_train_comps/n_threads;

  double *alphaV = new double[this->n_train_comps];
  double *alphaU = new double[this->n_train_comps];
  //memset(alphaU, 0, sizeof(double) * this->n_train_comps);
  //memset(alphaV, 0, sizeof(double) * this->n_train_comps);
  for(int i=0; i<n_train_comps; ++i) {
    alphaV[i] = .5/lambda;
    alphaU[i] = .5/lambda;
  }

  // Alternating RankSVM
  double start = omp_get_wtime();
  double f, f_old;

  initialize(init_option);

  f_old = compute_loss(model, train, loss_option) + .5 * lambda * (model.Unormsq() + model.Vnormsq());
  printf("0, %f, %f, %f, %f, %f, ", omp_get_wtime() - start, f_old, model.Unormsq(), model.Vnormsq(), compute_loss(model, train, loss_option));
  eval.evaluate(model);
  printf("\n");

  double normsq;
  for (int OuterIter = 1; OuterIter <= MaxIter; ++OuterIter) {

     
 
    ///////////////////////////
    // Learning V 
    ///////////////////////////
     
    // initialize using the previous alphaV
    memset(model.V, 0, sizeof(double) * n_items * model.rank);
//    memset(alphaV, 0, sizeof(double) * this->n_train_comps);
    
    #pragma omp parallel for
    for(int i=0; i<n_train_comps; ++i) {
      double *user_vec  = &(model.U[train[i].user_id  * model.rank]);
      double *item1_vec = &(model.V[train[i].item1_id * model.rank]);
      double *item2_vec = &(model.V[train[i].item2_id * model.rank]);
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
        double *user_vec  = &(model.U[train[idx].user_id  * model.rank]);
        double *item1_vec = &(model.V[train[idx].item1_id * model.rank]);
        double *item2_vec = &(model.V[train[idx].item2_id * model.rank]);
    
        double p1 = 0., p2 = 0., d = 0.;
        for(int j=0; j<model.rank; ++j) {
          d = item1_vec[j] - item2_vec[j];
          p1 += user_vec[j] * d;
          p2 += user_vec[j] * user_vec[j];
        } 

        double delta = dcd_delta(loss_option, alphaV[idx], p2*2., p1, 1./lambda);

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

    // compute performance measure
    f = compute_loss(model, train, loss_option) + .5*lambda*(model.Unormsq() + model.Vnormsq());
    printf("%d, %f, %f, %f, %f, %f, ", OuterIter, omp_get_wtime() - start, f, model.Unormsq(), model.Vnormsq(), compute_loss(model, train, loss_option));
    eval.evaluate(model);
    printf("\n");

    ///////////////////////////
    // Learning U 
    ///////////////////////////
     
    // initialize U using the previous alphaU 
    memset(model.U, 0, sizeof(double) * n_users * model.rank);
//    memset(alphaU, 0, sizeof(double) * this->n_train_comps);
    #pragma omp parallel for
    for(int i=0; i<n_train_comps; ++i) {
      //if (alphaU[i] > 1e-10) {
        double *user_vec  = &(model.U[train[i].user_id  * model.rank]);
        double *item1_vec = &(model.V[train[i].item1_id * model.rank]);
        double *item2_vec = &(model.V[train[i].item2_id * model.rank]);
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
      std::uniform_int_distribution<int> randidx(tridx[uid_from], tridx[uid_to]-1);

      for(int n_updates=0; n_updates<n_max_updates; ++n_updates) {
        int idx = randidx(gen);
        double *user_vec  = &(model.U[train[idx].user_id  * model.rank]);
        double *item1_vec = &(model.V[train[idx].item1_id * model.rank]);
        double *item2_vec = &(model.V[train[idx].item2_id * model.rank]);
    
        double p1 = 0., p2 = 0., d = 0.;
        for(int j=0; j<model.rank; ++j) {
          d = item1_vec[j] - item2_vec[j];
          p1 += user_vec[j] * d;
          p2 += d*d;
        } 

        double delta = dcd_delta(loss_option, alphaU[idx], p2, p1, 1./lambda);
        
        if (delta != 0.) {
          alphaU[idx] += delta;
          for(int j=0; j<model.rank; ++j) {
            d = delta * (item1_vec[j] - item2_vec[j]);
            user_vec[j] += d;
          }
        }
      }
		}

    // compute performance measure 
    f = compute_loss(model, train, loss_option) + .5*lambda*(model.Unormsq() + model.Vnormsq());
    printf("%d, %f, %f, %f, %f, %f, ", OuterIter, omp_get_wtime() - start, f, model.Unormsq(), model.Vnormsq(), compute_loss(model, train, loss_option));
    eval.evaluate(model);
    printf("\n");

    // stopping rule
    if ((f_old - f) / f_old < 1e-5) break;
    f_old = f;
  
  }

	delete [] alphaV;
	delete [] alphaU;
}	

void Problem::initialize(init_option_t option) {

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

    srand(time(NULL)); 
    for(int i=0; i<n_users*model.rank; i++) model.U[i] = (double)rand() / (double)RAND_MAX / sqrt((double)model.rank);
    for(int i=0; i<n_items*model.rank; i++) model.V[i] = (double)rand() / (double)RAND_MAX / sqrt((double)model.rank);
  }

}

#endif
