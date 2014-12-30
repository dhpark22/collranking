// Project: Parallel Collaborative Ranking with Alt-rankSVM and SGD
// Collaborative work by Dohyung Park and Jin Zhang
// Date: 11/26/2014
//
// The script will:
// [1]  convert preference data into item-based graph (adjacency matrix format)
// [2]  partition graph with Graclus
// [3a] solve the problem with alternative rankSVM via liblineaer
// [3b] solve the problem with stochasitic gradient descent in hogwild style
// [3c] solve the problem with stochastic gradient descent in nomad style
//
// Run: ./a.out [rating_file] [rating_format] [graph_output] [num_partitions]

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
#include "collrank.h"

using namespace std;

class Problem {
  protected:
    bool is_allocated, is_clustered;
    int n_users, n_items, n_train_comps, n_test_comps; 	// number of users/items in training sample, number of samples in traing and testing data set
    int rank, lambda, n_threads;                       // parameters
    double *U, *V;                                  // low rank U, V
    double alpha, beta;                             // parameter for sgd
    vector<int> n_comps_by_user, n_comps_by_item;

    vector<comparison>   train, train_item, test;
    vector<int>          tridx, tridx_item;

    bool sgd_step(const comparison&, const bool, const double, const double);
    void de_allocate();					            // deallocate U, V when they are used multiple times by different methods
  
  public:
    Problem(int, int);				// default constructor
    ~Problem();					// default destructor
    void read_data(char* train_file, char* test_file);	// read function
    void alt_rankSVM(double l);
    void run_sgd_random(double l, double a, double b);
    void run_sgd_nomad(double l, double a, double b);
    double compute_ndcg();
    double compute_testerror();
};

// may be more parameters can be specified here
Problem::Problem (int r, int np) {
	this->rank = r;
	this->is_allocated = false;
	this->is_clustered = false;
	this->n_threads = np;
}

Problem::~Problem () {
	//printf("calling the destructor\n");
	this->de_allocate();
}

void Problem::read_data(char* train_file, char* test_file) {

  n_users = n_items = 0;
  ifstream f(train_file);
  if (f) {
    int uid, i1id, i2id, uid_current = 0;
    tridx.push_back(0);
    while (f >> uid >> i1id >> i2id) {
      n_users = max(uid, n_users);
      n_items = max(i1id, max(i2id, n_items));
      --uid; --i1id; --i2id;

      if (uid_current < uid) {
        tridx.push_back(train.size());
        uid_current = uid;
      }

      train.push_back(comparison(uid, i1id, i2id, 1));	// now user and item starts from 0

      train_item.push_back(comparison(uid, i1id, i2id, 1));
      train_item.push_back(comparison(uid, i2id, i1id, -1));
    }
    tridx.push_back(train.size());
    n_train_comps = train.size();
  } else {
    printf("Error in opening the training file!\n");
    exit(EXIT_FAILURE);
  }
  f.close();

  vector<int> upart(n_threads+1);
  for(int tid=0; tid<=n_threads; ++tid) {
    upart[tid] = n_users * tid / n_threads;
  }

  tridx_item.resize(n_items * n_threads + 1);
  sort(train_item.begin(), train_item.end(), comp_itemwise);
 
  int idx = 0;
  for(int iid=0; iid<n_items; ++iid) {
    tridx_item[iid*n_threads] = idx;
    for(int tid=1; tid<=n_threads; ++tid) {
      while((train_item[idx].item1_id == iid) && (train_item[idx].user_id < upart[tid])) ++idx;
      tridx_item[iid*n_threads+tid] = idx;
    }
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

	n_comps_by_user.clear(); n_comps_by_user.resize(this->n_users);
	n_comps_by_item.clear(); n_comps_by_item.resize(this->n_items);
	for(int i=0; i<this->n_users; i++) n_comps_by_user[i] = 0;
	for(int i=0; i<this->n_items; i++) n_comps_by_item[i] = 0;

	for(int i=0; i<this->n_train_comps; i++) {
		++n_comps_by_user[train[i].user_id];
		++n_comps_by_item[train[i].item1_id];
		++n_comps_by_item[train[i].item2_id];
	}

	ifstream f2(test_file);
	if (f2) {
		int u, i, j;
		while (f2 >> u >> i >> j) {
			this->n_users = max(u, this->n_users);
			this->n_items = max(i, max(j, this->n_items));
			this->test.push_back(comparison(u-1, i-1, j-1, 1));
		}
		this->n_test_comps = this->test.size();
	} else {
		printf("error in opening the testing file\n");
		exit(EXIT_FAILURE);
	}
	f2.close();

  printf("%d users, %d items, %d training comps, %d test comps \n", this->n_users, this->n_items,
                                                                    this->n_train_comps,
                                                                    this->n_test_comps);

	this->U = new double [this->n_users * this->rank];
	this->V = new double [this->n_items * this->rank];

}	

void Problem::alt_rankSVM (double l) {

  lambda = l;

  int n_max_updates = n_train_comps/20/n_threads;

  double *alphaV = new double[this->n_train_comps];
  double *alphaU = new double[this->n_train_comps];
  memset(alphaU, 0, sizeof(double) * this->n_train_comps);
  memset(alphaV, 0, sizeof(double) * this->n_train_comps);
	
  // Alternating RankSVM
  for(int i=0; i<n_users*rank; ++i) U[i] = 1.;
  memset(V, 0, sizeof(double) * n_items * rank);
  printf("Initial test error : %f \n", this->compute_testerror());

  double start = omp_get_wtime(), error;
  for (int OuterIter = 0; OuterIter < 10; ++OuterIter) {
    // Learning V 
    memset(V, 0, sizeof(double) * n_items * rank);
    #pragma omp parallel for
    for(int i=0; i<n_train_comps; ++i) {
     if (alphaV[i] > 1e-10) {
        double *user_vec  = &U[train[i].user_id  * rank];
        double *item1_vec = &V[train[i].item1_id * rank];
        double *item2_vec = &V[train[i].item2_id * rank];
        for(int j=0; j<rank; ++j) {
          double d = alphaV[i] * user_vec[j];
          item1_vec[j] += d;
          item2_vec[j] -= d;  
        }
      }
    }		

    double p = 0.;
    #pragma omp parallel for reduction(+:p)
    for(int i=0; i<n_items*rank; ++i) { 
      double d = V[i]*V[i];
      p += d;
    }

    if (p > 1e-4) {
      p = sqrt(p);

      #pragma omp parallel for
      for(int i=0; i<n_items*rank; ++i) V[i] /= p;

      #pragma omp parallel for
      for(int i=0; i<n_train_comps; ++i) alphaV[i] /= p;
    }

    #pragma omp parallel
    {
      int i_thread = omp_get_thread_num();

      std::mt19937 gen(n_threads*OuterIter + i_thread);
      std::uniform_int_distribution<int> randidx(0, n_train_comps-1);

      for(int n_updates=0; n_updates<n_max_updates; ++n_updates) {
        int idx = randidx(gen);
        double *user_vec  = &U[train[idx].user_id  * rank];
        double *item1_vec = &V[train[idx].item1_id * rank];
        double *item2_vec = &V[train[idx].item2_id * rank];
    
        double p1 = 0., p2 = 0., d = 0.;
        for(int j=0; j<rank; ++j) {
          d = item1_vec[j] - item2_vec[j];
          p1 += user_vec[j] * d;
          p2 += user_vec[j] * user_vec[j];
        } 

        double delta = (1. - p1 - alphaV[idx]/2.*lambda) / (p2*2. + .5*lambda);
        delta = max(0., delta + alphaV[idx]) - alphaV[idx];      
 
        alphaV[idx] += delta;
        for(int j=0; j<rank; ++j) {
          item1_vec[j] += delta * user_vec[j]; 
          item2_vec[j] -= delta * user_vec[j]; 
        }
      }

		}

    error = this->compute_testerror();
	  printf("iteration %d, test error %f, time %f \n", OuterIter, error, omp_get_wtime() - start);
	
    // Learning U
    memset(U, 0, sizeof(double) * n_users * rank);
    #pragma omp parallel for
    for(int i=0; i<n_train_comps; ++i) {
     if (alphaU[i] > 1e-10) {
        double *user_vec  = &U[train[i].user_id  * rank];
        double *item1_vec = &V[train[i].item1_id * rank];
        double *item2_vec = &V[train[i].item2_id * rank];
        for(int j=0; j<rank; ++j) {
          user_vec[j] += alphaU[i] * (item1_vec[j] - item2_vec[j]);  
        }
      }
    }

    #pragma omp parallel for
    for(int i=0; i<n_users; ++i) {
      double p = 0.;
      int j = i*rank, j_end = (i+1)*rank; 
      for(; j<j_end; ++j) p += U[j]*U[j]; 
    
      if (p > 1e-4) {  
        p = sqrt(p);
        for(j=i*rank; j<j_end; ++j) U[j] /= p;    
        for(j=tridx[i]; j<tridx[i+1]; ++j) alphaU[j] /= p;
      }
    }

    #pragma omp parallel
    {
      int i_thread = omp_get_thread_num();
      int uid_from = (n_users * i_thread / n_threads);
      int uid_to   = (n_users * (i_thread+1) / n_threads);

      std::mt19937 gen(n_threads*OuterIter + i_thread);
      std::uniform_int_distribution<int> randidx(tridx[uid_from], tridx[uid_to]-1);

      for(int n_updates=0; n_updates<n_max_updates; ++n_updates) {
        int idx = randidx(gen);
        double *user_vec  = &U[train[idx].user_id  * rank];
        double *item1_vec = &V[train[idx].item1_id * rank];
        double *item2_vec = &V[train[idx].item2_id * rank];
    
        double p1 = 0., p2 = 0., d = 0.;
        for(int j=0; j<rank; ++j) {
          d = item1_vec[j] - item2_vec[j];
          p1 += user_vec[j] * d;
          p2 += d*d;
        } 

        double delta = (1. - p1 - alphaU[idx]*.5*lambda) / (p2 + .5*lambda);
        delta = max(0., delta + alphaU[idx]) - alphaU[idx];      
 
        alphaU[idx] += delta;
        for(int j=0; j<rank; ++j) user_vec[j] += delta * (item1_vec[j] - item2_vec[j]); 
      }
		}

    error = this->compute_testerror();
 	  printf("iteration %d, test error %f, time %f \n", OuterIter, error, omp_get_wtime() - start);

    if (OuterIter < 5) n_max_updates *= 3;	
  }

	delete [] alphaV;
	delete [] alphaU;
}	

bool Problem::sgd_step(const comparison& comp, const bool first_item_only, const double l, const double step_size) {
  double *user_vec  = &U[comp.user_id  * rank];
  double *item1_vec = &V[comp.item1_id * rank];
  double *item2_vec = &V[comp.item2_id * rank];

  int n_comps_user  = n_comps_by_user[comp.user_id];
  int n_comps_item1 = n_comps_by_item[comp.item1_id];
  int n_comps_item2 = n_comps_by_item[comp.item2_id];

  if ((n_comps_user < 1) || (n_comps_item1 < 1) || (n_comps_item2 < 1))
    printf("ERROR\n");

  double err = 1.;
  for(int k=0; k<rank; k++) err -= user_vec[k] * comp.comp * (item1_vec[k] - item2_vec[k]);

  if (err > 0) {	
    double grad = -2. * err;		// gradient direction for l2 hinge loss

    for(int k=0; k<rank; k++) {
	    double user_dir  = step_size * (grad * comp.comp * (item1_vec[k] - item2_vec[k]) + l / (double)n_comps_user * user_vec[k]);
	    double item1_dir = step_size * (grad * comp.comp * user_vec[k] + l / (double)n_comps_item1 * item1_vec[k]);
      double item2_dir;

      if (!first_item_only) item2_dir = step_size * (grad * -comp.comp * user_vec[k] + l / (double)n_comps_item2 * item2_vec[k]);

	    user_vec[k]  -= user_dir;
	    item1_vec[k] -= item1_dir;
      if (!first_item_only) item2_vec[k] -= item2_dir;
    }

	return true;
  }

  return false;
}

void Problem::run_sgd_random(double l, double a, double b) {

  auto real_rand = std::bind(std::uniform_real_distribution<double>(0,1), std::mt19937(time(NULL)));
  for(int i=0; i<n_users*rank; i++) U[i] = real_rand();
  for(int i=0; i<n_items*rank; i++) V[i] = real_rand();

  lambda = l;
  alpha  = a;
  beta   = b;

  int n_max_updates = n_train_comps/20/n_threads;

  printf("Initial test error : %f \n", this->compute_testerror());

  std::vector<int> c(n_train_comps,0);

  double time = omp_get_wtime();
  for(int icycle=0; icycle<20; ++icycle) {
    #pragma omp parallel
    {
      std::mt19937 gen(n_threads*icycle+omp_get_thread_num());
      std::uniform_int_distribution<int> randidx(0, n_train_comps-1);

      for(int n_updates=1; n_updates<n_max_updates; ++n_updates) {
        int idx = randidx(gen);
        ++c[idx];
        sgd_step(train[idx], false, lambda, 
                 alpha/(1.+beta*pow((double)((n_updates+n_max_updates*icycle)*n_threads),1.)));
      }
    }
/*
    // Normalize each row of U and the whole V
    #pragma omp parallel for
    for(int uid=0; uid<n_users; ++uid) {
      double p = 0.;
      for(int k=uid*rank; k<(uid+1)*rank; ++k) p += U[k]*U[k];
      if (p > 1e-4) {
        p = sqrt(p);
        for(int k=uid*rank; k<(uid+1)*rank; ++k) U[k] /= p;
      }
    }

    double p = 0.;
    #pragma omp parallel for reduction(+:p)
    for(int i=0; i<n_items*rank; ++i) {
      double d = V[i]*V[i];
      p += d;
    } 
    if (p > 1e-4) {
      p = sqrt(p);
      #pragma omp parallel for
      for(int i=0; i<n_items*rank; ++i) V[i] /= p;
    }
*/
    double error = this->compute_testerror();
    if (error < 0.) break; 
    printf("%d: iter %d, error %f, time %f \n", n_threads, (icycle+1)*n_max_updates, error, omp_get_wtime() - time);
  } 

}

void Problem::run_sgd_nomad(double l, double a, double b) {

  auto real_rand = std::bind(std::uniform_real_distribution<double>(0,1), std::mt19937(time(NULL)));
  for(int i=0; i<n_users*rank; i++) U[i] = real_rand();
  for(int i=0; i<n_items*rank; i++) V[i] = real_rand();

  lambda = l;
  alpha  = a;
  beta   = b;

  int n_max_updates = n_train_comps/5/n_threads;
  int queue_size = n_items+1;

  printf("Initial test error : %f \n", this->compute_testerror());

  int **queue = new int*[n_threads];
  std::vector<int> front(n_threads), back(n_threads);
  for(int i=0; i<n_threads; ++i) queue[i] = new int[queue_size];
   
  for(int i=0; i<n_threads; ++i) {
    for(int j=(n_items*i/n_threads), k=0; j<(n_items*(i+1)/n_threads); ++j, ++k) queue[i][k] = j;
    front[i] = 0;
    back[i]  = (n_items*(i+1)/n_threads) - (n_items*i/n_threads);
  }

  std::vector<int> c(n_train_comps,0);

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
            sgd_step(train_item[idx], false, lambda, 
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

    double error = this->compute_testerror();
    if (error < 0.) break; 
    printf("%d: iter %d, error %f, time %f \n", n_threads, n_updates_total, error, omp_get_wtime() - time);
  }

  for(int i=0; i<n_threads; i++) delete[] queue[i];
  delete[] queue;
}

double Problem::compute_ndcg() {
	double ndcg_sum = 0.;
	for(int i=0; i<n_users; i++) {
		double dcg = 0.;
		double norm = 1.;
		// compute dcg
		ndcg_sum += dcg / norm;
	}
}

double Problem::compute_testerror() {
	int n_error = 0; 

  return 0.;
    /* 
    for(int i=0; i<100; i++) {
        int idx = (int)((double)n_test_comps * (double)rand() / (double)RAND_MAX);
		double prod = 0.;
		int user_idx  = test[idx].user_id * rank;
		int item1_idx = test[idx].item1_id * rank;
		int item2_idx = test[idx].item2_id * rank;
		for(int k=0; k<rank; k++) prod += U[user_idx + k] * (V[item1_idx + k] - V[item2_idx + k]);
	    printf("%f ", i, prod);
    }
    printf("\n");
    */

  for(int i=0; i<n_users*rank; i++)
    if ((U[i] > 1e2) || (U[i] < -1e2)) {
      printf("U large number : %d %d %f \n", i/rank, i%rank, U[i]);   
      return -1.;
    }

  for(int i=0; i<n_items*rank; i++)
    if ((V[i] > 1e2) || (V[i] < -1e2)) {
      printf("V large number : %d %d %f \n", i/rank, i%rank, V[i]);   
      return -1.;
    }

//  for(int i=0; i<rank; ++i) printf("%f ", U[i]); printf("\n");
//  for(int i=0; i<rank; ++i) printf("%f ", V[i]); printf("\n");

	for(int i=0; i<n_test_comps; i++) {
		double prod = 0.;
		int user_idx  = test[i].user_id;
		int item1_idx = test[i].item1_id;
		int item2_idx = test[i].item2_id;
		for(int k=0; k<rank; k++) prod += U[user_idx*rank + k] * (V[item1_idx*rank + k] - V[item2_idx*rank + k]);
		if (prod <= 0.) n_error += 1;
		if (prod != prod) {
			printf("NaN detected %d %d %d %d %d \n", user_idx, item1_idx, item2_idx, n_comps_by_item[item1_idx], n_comps_by_item[item2_idx]);
            for(int k=0;k<rank; ++k) printf("%f %f %f \n", U[user_idx*rank + k], V[item1_idx*rank + k], V[item2_idx*rank + k]);
            return -1.;
        }
	}
  return (double)n_error / (double)n_test_comps;
}

void Problem::de_allocate () {
	delete [] this->U;
	delete [] this->V;
	this->U = NULL;
	this->V = NULL;
}

int main (int argc, char* argv[]) {
	if (argc < 4) {
		cout << "Solve collaborative ranking problem with given training/testing data set" << endl;
		cout << "Usage ./collaborative_ranking  : [training file] [testing file] [num_threads]" << endl;
		return 0;
	}

	int n_threads = atoi(argv[3]);

	Problem p(10, n_threads);		// rank = 10
	p.read_data(argv[1], argv[2]);
	omp_set_dynamic(0);
	omp_set_num_threads(n_threads);
	double start = omp_get_wtime();
  
  printf("Running AltSVM.. \n");  
	p.alt_rankSVM(1.);
	double m1 = omp_get_wtime() - start;
	printf("%d threads, rankSVM takes %f seconds until error %f \n", n_threads, m1, p.compute_testerror());
	
  printf("Running Random SGD.. \n");
  p.run_sgd_random(1., 1e-1, 1e-5);
	double m2 = omp_get_wtime() - start - m1;
  printf("%d threads, randSGD takes %f seconds until error %f \n", n_threads, m2, p.compute_testerror());

  printf("Running NOMAD SGD.. \n");
  p.run_sgd_nomad(1., 1e-1, 1e-5);
  double m3 = omp_get_wtime() - start - m2 - m1;
  printf("%d threads, nomadSGD takes %f seconds, error %f \n", n_threads, m3, p.compute_testerror());

  return 0;
}
