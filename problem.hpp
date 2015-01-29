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
    int n_threads;                          // parameters
    double lambda;
    double alpha, beta;                             // parameter for sgd
    vector<int> n_comps_by_user, n_comps_by_item;

    Model model;
    loss_option_t loss_option = L2_HINGE;

    vector<comparison>   train, train_user, train_item;
    vector<int>          tridx, tridx_user, tridx_item;

    RatingMatrix tr;

    double dcd_delta(loss_option_t, double, double, double, double);
    bool sgd_step(const comparison&, loss_option_t, double, double);
    void de_allocate();					            // deallocate U, V when they are used multiple times by different methods
    void initialize(init_option_t); 
    
    comparison random_comp(int, int);
    std::mt19937 random_comp_gen;
 
  public:
    Problem(int, int);				// default constructor
    ~Problem();					// default destructor
    void read_data(char*);	// read function
    void run_altsvm(Evaluator&, loss_option_t, double, init_option_t, int);
    void run_sgd_random(loss_option_t, double, double, double, init_option_t);
    void run_sgd_nomad_user(double, double, double, init_option_t);
    void run_sgd_nomad_item(double, double, double, init_option_t);
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
    int uid, i1id, i2id;
    while (f >> uid >> i1id >> i2id) {
      n_users = max(uid, n_users);
      n_items = max(i1id, max(i2id, n_items));
      --uid; --i1id; --i2id; // now user_id and item_id starts from 0

      train.push_back(comparison(uid, i1id, i2id, 1));

      train_user.push_back(comparison(uid, i1id, i2id, 1));
      train_user.push_back(comparison(uid, i2id, i1id, -1));
 
      train_item.push_back(comparison(uid, i1id, i2id, 1));
      train_item.push_back(comparison(uid, i2id, i1id, -1));
    }
    n_train_comps = train.size();
  } else {
    printf("Error in opening the training file!\n");
    exit(EXIT_FAILURE);
  }
  f.close();

  printf("Read %d training comparisons\n", n_train_comps);
  printf("%d users, %d items\n", n_users, n_items);

  n_comps_by_user.resize(this->n_users,0);
  n_comps_by_item.resize(this->n_items,0);

  // Construct tridx
  tridx.resize(n_users+1);
  sort(train.begin(), train.end(), comp_userwise);
  tridx[0] = 0;
  tridx[n_users] = n_train_comps;
  for(int idx=1; idx<n_train_comps; ++idx)
    if (train[idx-1].user_id < train[idx].user_id) tridx[train[idx].user_id] = idx;

  // Construct train_user structure
  vector<int> ipart(n_threads+1);
  for(int tid=0; tid<=n_threads; ++tid) {
    ipart[tid] = n_items * tid / n_threads;
  }

  tridx_user.resize(n_users * n_threads + 1);
  sort(train_user.begin(), train_user.end(), comp_userwise);
   
  int idx = 0;
  for(int uid=0; uid<n_users; ++uid) {
    tridx_user[uid*n_threads] = idx;
    for(int tid=1; tid<=n_threads; ++tid) {
      while((train_user[idx].user_id == uid) && (train_user[idx].item1_id < ipart[tid])) ++idx;
      tridx_user[uid*n_threads+tid] = idx;
    }
    n_comps_by_user[uid] = tridx_user[(uid+1)*n_threads] - tridx_user[uid*n_threads];
  }

  for(int uid=0; uid<n_users; ++uid) {
    for(int tid=0; tid<n_threads; ++tid) {
      for(int idx=tridx_user[uid*n_threads+tid]; idx<tridx_user[uid*n_threads+tid+1]; ++idx) {
        if (train_user[idx].user_id != uid) printf("ERROR indexing \n");
        if (train_user[idx].item1_id < ipart[tid]) printf("ERROR indexing \n");
        if (train_user[idx].item1_id >= ipart[tid+1]) printf("ERROR indexing \n");
      }
    }
  }

  // Construct train_item structure
  vector<int> upart(n_threads+1);
  for(int tid=0; tid<=n_threads; ++tid) {
    upart[tid] = n_users * tid / n_threads;
  }

  tridx_item.resize(n_items * n_threads + 1);
  sort(train_item.begin(), train_item.end(), comp_itemwise);
   
  idx = 0;
  for(int iid=0; iid<n_items; ++iid) {
    tridx_item[iid*n_threads] = idx;
    for(int tid=1; tid<=n_threads; ++tid) {
      while((train_item[idx].item1_id == iid) && (train_item[idx].user_id < upart[tid])) ++idx;
      tridx_item[iid*n_threads+tid] = idx;
    }
    n_comps_by_item[iid] = tridx_item[(iid+1)*n_threads] - tridx_item[iid*n_threads];
  }

  for(int iid=0; iid<n_items; ++iid) {
    for(int tid=0; tid<n_threads; ++tid) {
      for(int idx=tridx_item[iid*n_threads+tid]; idx<tridx_item[iid*n_threads+tid+1]; ++idx) {
        if (train_item[idx].item1_id != iid) printf("ERROR indexing \n");
        if (train_item[idx].user_id < upart[tid]) printf("ERROR indexing \n");
        if (train_item[idx].user_id >= upart[tid+1]) printf("ERROR indexing \n");
      }
    }
  }

  // reat train ratings (lsvm format)
  //tr.read_lsvm(train_rating);

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

void Problem::run_altsvm(Evaluator& eval, loss_option_t loss_option = L2_HINGE, double l = 10., init_option_t init_option = INIT_RANDOM, int MaxIter = 50) {

  printf("Alternating rankSVM with %d threads.. \n", n_threads);

  lambda = l;
  double lambda_user = l / n_users * n_items;
  double lambda_item = l;

  int n_max_updates = n_train_comps/n_threads;

  double *alphaV = new double[this->n_train_comps];
  double *alphaU = new double[this->n_train_comps];
  memset(alphaU, 0, sizeof(double) * this->n_train_comps);
  memset(alphaV, 0, sizeof(double) * this->n_train_comps);

  double *slack  = new double[this->n_train_comps];
  memset(slack,  0, sizeof(double) * this->n_train_comps);
    
  // Alternating RankSVM
  double start = omp_get_wtime();
  double f, f_old;

  initialize(init_option);

  f_old = compute_loss(model, train, loss_option) + .5 * lambda * (model.Unormsq() + model.Vnormsq());
  printf("0, %f / %f, %f, %f, %f", omp_get_wtime() - start, f_old, model.Unormsq(), model.Vnormsq(), compute_loss(model, train, loss_option));
  eval.evaluate(model);
  printf("\n");

  double normsq;
  for (int OuterIter = 1; OuterIter <= MaxIter; ++OuterIter) {
      
    ///////////////////////////
    // Learning V 
    ///////////////////////////
     
    // initialize using the previous alphaV
    memset(model.V, 0, sizeof(double) * n_items * model.rank);
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
    printf("%d, %f / %f, %f, %f, %f", OuterIter, omp_get_wtime() - start, f, model.Unormsq(), model.Vnormsq(), compute_loss(model, train, loss_option));
    eval.evaluate(model);
    printf("\n");


    ///////////////////////////
    // Learning U 
    ///////////////////////////
     
    // initialize U using the previous alphaU 
    memset(model.U, 0, sizeof(double) * n_users * model.rank);
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

        alphaU[idx] += delta;
        for(int j=0; j<model.rank; ++j) {
          d = delta * (item1_vec[j] - item2_vec[j]);
          user_vec[j] += d;
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

  delete [] slack;
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

    int user_idx, item_idx;
 
    for(int iter=0; iter<100; ++iter) {
      //printf("%d \n", iter);

      // normalize U (Gram-Schmidt)
      for(int k=0; k<model.rank; ++k) {
      
        double normsq = 0.;
        for(int i=k; i<n_users*model.rank; i+=model.rank) normsq += model.U[i]*model.U[i];

        double norm = sqrt(normsq);
        for(int i=k; i<n_users*model.rank; i+=model.rank) model.U[i] /= norm;

        for(int j=1; j<model.rank-k; ++j) {
          double dotprod = 0.;
          for(int i=k; i<n_users*model.rank; i+=model.rank) dotprod += model.U[i] * model.U[i+j];
          for(int i=k; i<n_users*model.rank; i+=model.rank) model.U[i+j] -= dotprod * model.U[i];
        }
      }    
  
      // left multiplication with U
      memset(model.V, 0, sizeof(double) * n_items * model.rank);
      for(int iid=0; iid<n_items; ++iid) {
        item_idx = iid * model.rank;
        for(int i=tridx_item[iid*n_threads]; i<tridx_item[(iid+1)*n_threads]; ++i) {
          user_idx = train_item[i].user_id * model.rank;
          for(int k=0; k<model.rank; ++k) model.V[item_idx+k] += model.U[user_idx+k] * train_item[i].comp;
        }
      }
 
      // normalize V (Gram-Schmidt)
      for(int k=0; k<model.rank; ++k) {
      
        double normsq = 0.;
        for(int i=k; i<n_items*model.rank; i+=model.rank) normsq += model.V[i]*model.V[i];

        double norm = sqrt(normsq);
        for(int i=k; i<n_items*model.rank; i+=model.rank) model.V[i] /= norm;
     
        for(int j=1; j<model.rank-k; ++j) {
          double dotprod = 0.;
          for(int i=k; i<n_items*model.rank; i+=model.rank) dotprod += model.V[i] * model.V[i+j];
          for(int i=k; i<n_items*model.rank; i+=model.rank) model.V[i+j] -= dotprod * model.V[i];
        }
      
      }    

      // right multiplication with V
      memset(model.U, 0, sizeof(double) * n_users * model.rank);
      for(int uid=0; uid<n_users; ++uid) {
        user_idx = uid * model.rank;
        for(int i=tridx_user[uid*n_threads]; i<tridx_user[(uid+1)*n_threads]; ++i) {
          item_idx = train_user[i].item1_id * model.rank;
          for(int k=0; k<model.rank; ++k) model.U[user_idx+k] += model.V[item_idx+k] * train_user[i].comp;
        }
      }

    }

    double norm;
    
    for(int i=0; i<n_users; ++i) {
      norm = 0.;
      user_idx = i * model.rank;
      for(int k=0; k<model.rank; ++k) norm += model.U[user_idx+k]*model.U[user_idx+k];
      if (norm > 1e-6) {
        norm = sqrt(norm);
        for(int k=0; k<model.rank; ++k) model.U[user_idx+k] /= norm;
      }
    }
  }

}


/*
bool Problem::sgd_step(const comparison& comp, loss_option_t loss_option, double l, double step_size) {
  double *user_vec  = &(model.U[comp.user_id  * model.rank]);
  double *item1_vec = &(model.V[comp.item1_id * model.rank]);
  double *item2_vec = &(model.V[comp.item2_id * model.rank]);

  int n_comps_user  = n_comps_by_user[comp.user_id];
  int n_comps_item1 = n_comps_by_item[comp.item1_id];
  int n_comps_item2 = n_comps_by_item[comp.item2_id];

  if ((n_comps_user < 1) || (n_comps_item1 < 1) || (n_comps_item2 < 1)) printf("ERROR\n");

  double prod = 0.;
  for(int k=0; k<model.rank; k++) prod += user_vec[k] * comp.comp * (item1_vec[k] - item2_vec[k]);

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

	  return true;
  }

  return false;
}

void Problem::run_sgd_random(loss_option_t loss_option = L2_HINGE, double l = 10., double a = 1., double b = 1., init_option_t option = INIT_RANDOM) {

  printf("Random SGD with %d threads..\n", n_threads);

  double time = omp_get_wtime();
  this->initialize(option); 

  std::pair<double,double> error;
  double ndcg, f;
  error = compute_pairwiseError(test, model);
  ndcg  = compute_ndcg(test, model);
  f = compute_loss(model, train, loss_option) + .5*lambda*(model.Unormsq() + model.Vnormsq());
  printf("0, %f, %f, %f, %f / %f, %f, %f / %f\n", f, model.Unormsq(), model.Vnormsq(), compute_loss(model, train, loss_option), error.first, error.second, ndcg, omp_get_wtime() - time);

  lambda = l;
  alpha  = a;
  beta   = b;

  int n_max_updates = n_train_comps/1000/n_threads;

  std::vector<int> c(n_train_comps,0);

  for(int icycle=0; icycle<20; ++icycle) {
    #pragma omp parallel
    {
      std::mt19937 gen(n_threads*icycle+omp_get_thread_num());
      std::uniform_int_distribution<int> randidx(0, n_train_comps-1);

      for(int n_updates=1; n_updates<n_max_updates; ++n_updates) {
        int idx = randidx(gen);
        ++c[idx];
        double stepsize = alpha/(1.+beta*pow((double)((n_updates+n_max_updates*icycle)*n_threads), .5));
        sgd_step(train[idx], loss_option, lambda, stepsize);
      }
    }

    error = compute_pairwiseError(test, model);
    ndcg  = compute_ndcg(test, model);
    f = compute_loss(model, train, loss_option) + .5*lambda*(model.Unormsq() + model.Vnormsq());
    printf("%d, %f, %f, %f, %f / %f, %f, %f / %f\n", (icycle+1)*n_max_updates, f, model.Unormsq(), model.Vnormsq(), compute_loss(model, train, loss_option), error.first, error.second, ndcg, omp_get_wtime() - time);
  
    if (icycle < 5) n_max_updates *= 4;
  } 

}

void Problem::run_sgd_nomad_user(double l, double a, double b, init_option_t option) {

  printf("NOMAD SGD-user with $d threads..\n", n_threads);

  this->initialize(option);

  lambda = l;
  alpha  = a;
  beta   = b;

  int n_max_updates = n_train_comps/1000/n_threads;
  int queue_size = n_users+1;

  int **queue = new int*[n_threads];
  std::vector<int> front(n_threads), back(n_threads);
  for(int i=0; i<n_threads; ++i) queue[i] = new int[queue_size];
   
  for(int i=0; i<n_threads; ++i) {
    for(int j=(n_users*i/n_threads), k=0; j<(n_users*(i+1)/n_threads); ++j, ++k) queue[i][k] = j;
    front[i] = 0;
    back[i]  = (n_users*(i+1)/n_threads) - (n_users*i/n_threads);
  }

  std::vector<int> c(n_train_comps*2,0);

  int n_updates_total = 0;

  double time = omp_get_wtime();
  for(int icycle=0; icycle<20; ++icycle) {
 		
    int flag = -1;

    #pragma omp parallel shared(n_updates_total, flag, queue, front, back)
    {
      std::mt19937 gen(n_threads*icycle+omp_get_thread_num());
      std::uniform_int_distribution<int> randtid(0, n_threads-1);

      int tid = omp_get_thread_num();
      int tid_next = tid-1;
      //if (tid_next < 0) tid_next = n_threads-1;

      int n_updates = 1;

      //printf("thread %d/%d beginning : users %d - %d  \n", tid, tid_next, queue[tid][front[tid]], queue[tid][back[tid]-1]);
      while((flag == -1) && (n_updates < n_max_updates)) {
        if (front[tid] != back[tid]) {
                
          int uid;
                
          //#pragma omp critical
          {
            uid = queue[tid][front[tid]];
            front[tid] = (front[tid]+1) % queue_size;
          }

          for(int idx=tridx_user[uid*n_threads+tid]; idx<tridx_user[uid*n_threads+tid+1]; ++idx) {
            sgd_step(train_user[idx], L2_HINGE, lambda, 
                     alpha/(1.+beta*(double)(n_updates_total + n_updates*n_threads)));
            ++n_updates;
          }

          tid_next = randtid(gen);
          #pragma omp critical
          {
            queue[tid_next][back[tid_next]] = uid;
            back[tid_next] = (back[tid_next]+1) % queue_size;
          }
        }
        else {
          flag = tid;
        }
	    }

	    if (flag == -1) flag = tid;

      #pragma omp atomic
      n_updates_total += n_updates;

    }

    std::pair<double,double> error = compute_pairwiseError(test, model);
    double ndcg  = compute_ndcg(test, model);
    printf("%d, %f, %f, %f, %f, %f, %f, %f\n", n_updates_total, model.Unormsq(), model.Vnormsq(), compute_loss(model, train, loss_option), error.first, error.second, ndcg, omp_get_wtime() - time);

    if (icycle < 5) n_max_updates *= 4;

  }

  for(int i=0; i<n_threads; i++) delete[] queue[i];
  delete[] queue;
}

void Problem::run_sgd_nomad_item(double l, double a, double b, init_option_t option) {

  printf("NOMAD SGD with $d threads..\n", n_threads);

  this->initialize(option);

  lambda = l;
  alpha  = a;
  beta   = b;

  int n_max_updates = n_train_comps/1000/n_threads;
  int queue_size = n_items+1;

  int **queue = new int*[n_threads];
  std::vector<int> front(n_threads), back(n_threads);
  for(int i=0; i<n_threads; ++i) queue[i] = new int[queue_size];
   
  for(int i=0; i<n_threads; ++i) {
    for(int j=(n_items*i/n_threads), k=0; j<(n_items*(i+1)/n_threads); ++j, ++k) queue[i][k] = j;
    front[i] = 0;
    back[i]  = (n_items*(i+1)/n_threads) - (n_items*i/n_threads);
  }

  std::vector<int> c(n_train_comps*2,0);

  int n_updates_total = 0;

  double time = omp_get_wtime();
  for(int icycle=0; icycle<20; ++icycle) {
 		
    int flag = -1;

    #pragma omp parallel shared(n_updates_total, flag, queue, front, back)
    {
      std::mt19937 gen(n_threads*icycle+omp_get_thread_num());
      std::uniform_int_distribution<int> randtid(0, n_threads-1);

      int tid = omp_get_thread_num();
      int tid_next = tid-1;
      //if (tid_next < 0) tid_next = n_threads-1;

      int n_updates = 1;

      //printf("thread %d/%d beginning : users %d - %d  \n", tid, tid_next, queue[tid][front[tid]], queue[tid][back[tid]-1]);
      while((flag == -1) && (n_updates < n_max_updates)) {
        if (front[tid] != back[tid]) {
                
          int iid;
                
          //#pragma omp critical
          {
            iid = queue[tid][front[tid]];
            front[tid] = (front[tid]+1) % queue_size;
          }

          for(int idx=tridx_item[iid*n_threads+tid]; idx<tridx_item[iid*n_threads+tid+1]; ++idx) {
            sgd_step(train_item[idx], L2_HINGE, lambda, 
                     alpha/(1.+beta*(double)(n_updates_total + n_updates*n_threads)));
            ++n_updates;
          }

          tid_next = randtid(gen);
          #pragma omp critical
          {
            queue[tid_next][back[tid_next]] = iid;
            back[tid_next] = (back[tid_next]+1) % queue_size;
          }
        }
        else {
          flag = tid;
        }
	    }

	    if (flag == -1) flag = tid;

      #pragma omp atomic
      n_updates_total += n_updates;

    }

    std::pair<double,double> error = compute_pairwiseError(test, model);
    double ndcg  = compute_ndcg(test, model);
    printf("%d, %f, %f, %f, %f, %f, %f, %f\n", n_updates_total, model.Unormsq(), model.Vnormsq(), compute_loss(model, train, loss_option), error.first, error.second, ndcg, omp_get_wtime() - time);

    if (icycle < 5) n_max_updates *= 4;

  }

  for(int i=0; i<n_threads; i++) delete[] queue[i];
  delete[] queue;
}
*/

#endif
