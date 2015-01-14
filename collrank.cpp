// Project: Parallel Collaborative Ranking with AltSVM and SGD
// Collaborative work by Dohyung Park and Jin Zhang
// Date: 11/26/2014
//
// The script will:
// [1a] solve the problem with alternative rankSVM via liblineaer
// [1b] solve the problem with stochasitic gradient descent in hogwild style
// [1c] solve the problem with stochastic gradient descent in nomad style
//
// Run: ./a.out [rating_file] [rating_format] [num_partitions]

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
#include "collrank.h"

using namespace std;

enum init_option_t {INIT_RANDOM, INIT_SVD};

class Problem {
  protected:
    bool is_allocated;
    int n_users, n_items, n_train_comps, n_test_comps; 	// number of users/items in training sample, number of samples in traing and testing data set
    int rank, lambda, n_threads;                       // parameters
    double *U, *V;                                  // low rank U, V
    double alpha, beta;                             // parameter for sgd
    vector<int> n_comps_by_user, n_comps_by_item;

    vector<comparison>   train, train_user, train_item, test;
    vector<int>          tridx, tridx_user, tridx_item;

    vector<rating>       ratings;
    vector<int>          rtidx;
    vector<double>       dcg_max;
    int ndcg_k = 10;

    bool sgd_step(const comparison&, const bool, const double, const double);
    void de_allocate();					            // deallocate U, V when they are used multiple times by different methods
    void initialize(init_option_t); 
 
  public:
    Problem(int, int);				// default constructor
    ~Problem();					// default destructor
    void read_data(char*, char*, char*);	// read function
    void run_altsvm(double, init_option_t);
    void run_sgd_random(double, double, double, init_option_t);
    void run_sgd_nomad_user(double, double, double, init_option_t);
    void run_sgd_nomad_item(double, double, double, init_option_t);
    double compute_loss();
    double compute_ndcg();
    double compute_testerror();
  };

  // may be more parameters can be specified here
  Problem::Problem (int r, int np) {
    this->rank = r;
    this->is_allocated = false;
    this->n_threads = np;
  }

  Problem::~Problem () {
    //printf("calling the destructor\n");
    this->de_allocate();
  }

  void Problem::read_data(char* test_ratings_file, char* train_file, char* test_file) {

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

    // Read test comparisons
    f.open(test_file);
    if (f.is_open()) {
      int uid, i1id, i2id;
      while (f >> uid >> i1id >> i2id) {
        n_users = max(uid, n_users);
        n_items = max(i1id, max(i2id, n_items));
        --uid; --i1id; --i2id;

        test.push_back(comparison(uid, i1id, i2id, 1));
      }
      n_test_comps = test.size();
    } else {
      printf("Error in opening the testing file!\n");
      exit(EXIT_FAILURE);
    }
    f.close();

    printf("Read %d training comparisons and %d test comparisons\n", n_train_comps, n_test_comps);
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

    // Read ratings file for NDCG
    vector<rating> ratings_current_user(0);
    string user_str, attribute_str;
    stringstream attribute_sstr;  
 
    dcg_max.resize(n_users,0.);
    f.open(test_ratings_file);
    if (f.is_open()) {
      int uid = 0, iid, score, uid_current = 0;
      rtidx.push_back(0);
      
      while(1) {
        getline(f, user_str);

        size_t pos1 = 0, pos2;
        while(1) {
          pos2 = user_str.find(':', pos1); if (pos2 == string::npos) break; 
          attribute_str = user_str.substr(pos1, pos2-pos1);
          attribute_sstr.clear(); attribute_sstr.str(attribute_str);
          attribute_sstr >> iid; --iid;
          pos1 = pos2+1;

          pos2 = user_str.find(' ', pos1); attribute_str = user_str.substr(pos1, pos2-pos1);
          attribute_sstr.clear(); attribute_sstr.str(attribute_str);
          attribute_sstr >> score;
          pos1 = pos2+1;

          ratings_current_user.push_back(rating(uid, iid, score));
        }
        if (ratings_current_user.size() == 0) break;
          
        for(int j=0; j<ratings_current_user.size(); ++j) ratings.push_back(ratings_current_user[j]);
        rtidx.push_back(ratings.size());

        sort(ratings_current_user.begin(), ratings_current_user.end(), comp_ratingwise);
        for(int k=0; k<ndcg_k; ++k) {
          dcg_max[uid] += (double)((1<<ratings_current_user[k].score) - 1) / log2(k+2); 
          if (uid == 0) printf("%d:%d ",ratings_current_user[k].item_id,ratings_current_user[k].score);
        }
        if (uid == 0) printf("\n");
    
        if (dcg_max[uid] < .1) {
          printf("%d %f : ", uid, dcg_max[uid]);
          for(int j=0; j<ratings_current_user.size(); ++j) printf("%d:%d ", ratings_current_user[j].item_id, ratings_current_user[j].score);
          printf("\n");
        }
 
        ratings_current_user.clear();
 
        ++uid;
      }

    } else {
      printf("Error in opening the extracted rating file!\n");
      exit(EXIT_FAILURE);
    }
    f.close();

    // Allocate memory for U and V
    this->U = new double [this->n_users * this->rank];
    this->V = new double [this->n_items * this->rank];

  }	

  double Problem::compute_loss() {
    double p = 0., slack;
    for(int i=0; i<n_train_comps; ++i) {
      double *user_vec  = &U[train[i].user_id  * rank];
      double *item1_vec = &V[train[i].item1_id * rank];
      double *item2_vec = &V[train[i].item2_id * rank];
      double d = 0.;
      for(int j=0; j<rank; ++j) {
        d += user_vec[j] * (item1_vec[j] - item2_vec[j]);
      }
      slack = max(0., 1. - d);
      p += slack*slack/lambda;
    }
    
    return p;		
  }

void Problem::run_altsvm(double l, init_option_t option) {

  printf("Alternating rankSVM with %d threads.. \n", n_threads);

  lambda = l;

  int n_max_updates = n_train_comps*10/n_threads;

  double *alphaV = new double[this->n_train_comps];
  double *alphaU = new double[this->n_train_comps];
  memset(alphaU, 0, sizeof(double) * this->n_train_comps);
  memset(alphaV, 0, sizeof(double) * this->n_train_comps);

  double *slack  = new double[this->n_train_comps];
  memset(slack,  0, sizeof(double) * this->n_train_comps);
    
  // Alternating RankSVM
  for(int i=0; i<n_users*rank; ++i) U[i] = 1.;
  memset(V, 0, sizeof(double) * n_items * rank);
    
  //initialize(INIT_RANDOM);
  //printf("Initial test error : %f \n", this->compute_testerror());

  double start = omp_get_wtime(), error, ndcg;
  for (int OuterIter = 0; OuterIter < 5; ++OuterIter) {
      
    ///////////////////////////
    // Learning V 
    ///////////////////////////
      
    // initialize using the previous alphaV
    memset(V, 0, sizeof(double) * n_items * rank);
    #pragma omp parallel for
    for(int i=0; i<n_train_comps; ++i) {
      double *user_vec  = &U[train[i].user_id  * rank];
      double *item1_vec = &V[train[i].item1_id * rank];
      double *item2_vec = &V[train[i].item2_id * rank];
      if (alphaV[i] > 1e-10) {
        for(int j=0; j<rank; ++j) {
          double d = alphaV[i] * user_vec[j];
          item1_vec[j] += d;
          item2_vec[j] -= d;
        }
      }
    }		

    // compute primal objective
    double normsq = 0.;
    #pragma omp parallel for reduction(+:normsq)
    for(int i=0; i<n_items*rank; ++i) { 
      double d = V[i]*V[i];
      normsq += d;
    }
    printf("primal f : %f -> ", .5*normsq + compute_loss());

/*
    // normalize V
    if (normsq > 1e-4) {
      double norm = sqrt(normsq);

      #pragma omp parallel for
      for(int i=0; i<n_items*rank; ++i) V[i] /= norm;

      #pragma omp parallel for
      for(int i=0; i<n_train_comps; ++i) alphaV[i] /= norm;
    }
*/

    // DUAL COORDINATE DESCENT for V
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

        if (delta != 0.) { 
          alphaV[idx] += delta;
          for(int j=0; j<rank; ++j) {
            d = delta * user_vec[j];
            item1_vec[j] += d; 
            item2_vec[j] -= d;
          }
        }
      }
    }

    // compute primal objective
    normsq = 0.;
    #pragma omp parallel for reduction(+:normsq)
    for(int i=0; i<n_items*rank; ++i) { 
      double d = V[i]*V[i];
      normsq += d;
    }
    printf("%f \n", .5*normsq + compute_loss());

    // compute error and ndcg
    error = compute_testerror();
    ndcg  = compute_ndcg();
    printf("%d, %f, %f, %f \n", OuterIter, error, ndcg, omp_get_wtime() - start);


    ///////////////////////////
    // Learning U 
    ///////////////////////////
     
    // initialize U using the previous alphaU 
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
/*
      // normalize U
    #pragma omp parallel for
    for(int uid=0; uid<n_users; ++uid) {
      double p = 0.;
      int j = uid*rank, j_end = (uid+1)*rank; 
      for(; j<j_end; ++j) p += U[j]*U[j]; 
    
      if (p > 1e-4) {  
        p = sqrt(p);
        for(j=uid*rank; j<j_end; ++j) U[j] /= p;    
        for(j=tridx_user[uid*n_threads]; j<tridx_user[(uid+1)*n_threads]; ++j) alphaU[j] /= p;
      }
    }
*/

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
        delta = max(0., alphaU[idx] + delta) - alphaU[idx];      
 
        alphaU[idx] += delta;
        for(int j=0; j<rank; ++j) {
          d = delta * (item1_vec[j] - item2_vec[j]);
          user_vec[j] += d;
        }
      }
		}

    // compute error and ndcg
    error = this->compute_testerror();
    ndcg  = this->compute_ndcg();
 	  printf("%d, %f, %f, %f \n", OuterIter, error, ndcg, omp_get_wtime() - start);

    if (OuterIter < 5) n_max_updates *= 2;	
  }

  delete [] slack;
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

void Problem::initialize(init_option_t option) {

  switch(option) {
    case INIT_RANDOM:
   
    srand(time(NULL)); 
    for(int i=0; i<n_users*rank; i++) U[i] = (double)rand() / (double)RAND_MAX / sqrt((double)rank);
    for(int i=0; i<n_items*rank; i++) V[i] = (double)rand() / (double)RAND_MAX / sqrt((double)rank);
    break;
    
    case INIT_SVD:

    srand(time(NULL)); 
    for(int i=0; i<n_users*rank; i++) U[i] = (double)rand() / (double)RAND_MAX / sqrt((double)rank);

    int user_idx, item_idx;
 
    for(int iter=0; iter<10; ++iter) {
      printf("%d \n", iter);

      // normalize U (Gram-Schmidt)
      for(int k=0; k<rank; ++k) {
      
        double normsq = 0.;
        for(int i=k; i<n_users*rank; i+=rank) normsq += U[i]*U[i];

        double norm = sqrt(normsq);
        for(int i=k; i<n_users*rank; i+=rank) U[i] /= norm;

        for(int j=1; j<rank-k; ++j) {
          double dotprod = 0.;
          for(int i=k; i<n_users*rank; i+=rank) dotprod += U[i]*U[i+j];
          for(int i=k; i<n_users*rank; i+=rank) U[i+j] -= dotprod*U[i];
        }
      }    
  
      // left multiplication with U
      //memset(V, 0, sizeof(double) * n_items * rank);
      for(int i=0; i<n_items*rank; ++i) V[i] = 0.;
      for(int iid=0; iid<n_items; ++iid) {
        item_idx = iid * rank;
        for(int i=tridx_item[iid*n_threads]; i<tridx_item[(iid+1)*n_threads]; ++i) {
          user_idx = train_item[i].user_id * rank;
          for(int k=0; k<rank; ++k) V[item_idx+k] += U[user_idx+k] * train_item[i].comp;
        }
      }
 
      // normalize V (Gram-Schmidt)
      for(int k=0; k<rank; ++k) {
      
        double normsq = 0.;
        for(int i=k; i<n_items*rank; i+=rank) normsq += V[i]*V[i];

        double norm = sqrt(normsq);
        for(int i=k; i<n_items*rank; i+=rank) V[i] /= norm;
     
        for(int j=1; j<rank-k; ++j) {
          double dotprod = 0.;
          for(int i=k; i<n_items*rank; i+=rank) dotprod += V[i]*V[i+j];
          for(int i=k; i<n_items*rank; i+=rank) V[i+j] -= dotprod*V[i];
        }
      
      }    

      // right multiplication with V
      //memset(U, 0, sizeof(double) * n_users * rank);
      for(int i=0; i<n_users*rank; ++i) U[i] = 0.;
      for(int uid=0; uid<n_users; ++uid) {
        user_idx = uid * rank;
        for(int i=tridx_user[uid*n_threads]; i<tridx_user[(uid+1)*n_threads]; ++i) {
          item_idx = train_user[i].item1_id * rank;
          for(int k=0; k<rank; ++k) U[user_idx+k] += V[item_idx+k] * train_user[i].comp;
        }
      }

    }

    double norm;
    
    for(int i=0; i<n_users; ++i) {
      norm = 0.;
      user_idx = i * rank;
      for(int k=0; k<rank; ++k) norm += U[user_idx+k]*U[user_idx+k];
      if (norm > 1e-6) {
        norm = sqrt(norm);
        for(int k=0; k<rank; ++k) U[user_idx+k] /= norm;
      }
    }
/*
    norm = 0.;
    for(int i=0; i<n_items*rank; ++i) norm += V[i]*V[i];
    if (norm > 1e-10) {
      norm = sqrt(norm) / sqrt((double)n_items);
      for(int i=0; i<n_items*rank; ++i) V[i] /= norm;
    }
*/
  }

}

void Problem::run_sgd_random(double l, double a, double b, init_option_t option) {

  printf("Random SGD with %d threads..\n", n_threads);

  printf("Initialize..\n");
  this->initialize(option); 

  printf("Initial error : ");
  double ndcg = this->compute_ndcg();
  double error = this->compute_testerror();
  printf("%f, %f \n", error, ndcg);
 
  printf("Running SGD..\n");

  lambda = l;
  alpha  = a;
  beta   = b;

  int n_max_updates = n_train_comps/1000/n_threads;

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
    double ndcg = this->compute_ndcg();
    double error = this->compute_testerror();
    if (error < 0.) break; 
    printf("%d, %f, %f, %f \n", (icycle+1)*n_max_updates, error, ndcg, omp_get_wtime() - time);
  
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
            sgd_step(train_user[idx], false, lambda, 
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

    double error = this->compute_testerror();
    double ndcg = this->compute_ndcg();
    if (error < 0.) break; 
    printf("%d, %f, %f, %f \n", n_updates_total, error, ndcg, omp_get_wtime() - time);

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
    double ndcg = this->compute_ndcg();
    if (error < 0.) break; 
    printf("%d, %f, %f, %f \n", n_updates_total, error, ndcg, omp_get_wtime() - time);

    if (icycle < 5) n_max_updates *= 4;

  }

  for(int i=0; i<n_threads; i++) delete[] queue[i];
  delete[] queue;
}

double Problem::compute_ndcg() {
	double ndcg_sum = 0.;
	vector<pair<double,int> > ranking;
 
  for(int uid=0; uid<n_users; ++uid) {
		double dcg = 0.;
  
    ranking.clear();
    for(int idx=rtidx[uid]; idx<rtidx[uid+1]; ++idx) {
      double prod = 0.;
      for(int k=0; k<rank; ++k) prod += U[uid * rank + k] * V[ratings[idx].item_id * rank + k];
      ranking.push_back(pair<double,int>(prod,0));
    }

    double min_score = ranking[0].first;
    for(int j=0; j<ranking.size(); ++j) min_score = min(min_score, ranking[j].first);

    for(int k=1; k<=ndcg_k; ++k) {
      int topk_idx = -1;
      double max_score = min_score - 1.;
      for(int j=0; j<ranking.size(); ++j) {
        if ((ranking[j].second == 0) && (ranking[j].first > max_score)) {
          max_score = ranking[j].first;
          topk_idx = j;
        }
      }
      ranking[topk_idx].second = k;
     
      dcg += (double)((1<<ratings[rtidx[uid]+topk_idx].score) - 1) / log2((double)(k+1));
    }
   
/*
    if (dcg/dcg_max[uid] < .8) {
      for(int j=0; j<ranking.size(); ++j) printf("%d:%d:%f:%d ", ratings[rtidx[uid]+j].item_id, ratings[rtidx[uid]+j].score, ranking[j].first, ranking[j].second); 
      printf("\n %d %f %f\n", uid, dcg, dcg_max[uid]);
    } 
*/

    ndcg_sum += dcg / dcg_max[uid];
	}

  return ndcg_sum / (double)n_users;
}

double Problem::compute_testerror() {
	int n_error = 0; 

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
/*
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
*/
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
	if (argc < 5) {
		cout << "Solve collaborative ranking problem with given training/testing data set" << endl;
		cout << "Usage ./collaborative_ranking [rating file] [training file] [testing file] [num_threads]" << endl;
		return 0;
	}

	int n_threads = atoi(argv[4]);

	Problem p(100, n_threads);		// rank = 10

  printf("Reading data files..\n");

	p.read_data(argv[1], argv[2], argv[3]);
	omp_set_dynamic(0);
	omp_set_num_threads(n_threads);
	
  double time;
 
  time = omp_get_wtime(); 
  printf("Running AltSVM.. \n");  
	p.run_altsvm(1., INIT_RANDOM);
	printf("%d threads, rankSVM takes %f seconds until error %f \n", n_threads, omp_get_wtime() - time, p.compute_testerror());

  time = omp_get_wtime();
  printf("Running Random SGD with random init.. \n");
  p.run_sgd_random(100., 1e-1, 1e-5, INIT_RANDOM);
  printf("%d threads, randSGD takes %f seconds until error %f \n", n_threads, omp_get_wtime() - time, p.compute_testerror());

/*
  time = omp_get_wtime();
  printf("Running Random SGD with SVD init.. \n");
  p.run_sgd_random(100., 1e-1, 1e-5, INIT_SVD);
  printf("%d threads, randSGD takes %f seconds until error %f \n", n_threads, omp_get_wtime() - time, p.compute_testerror());

  time = omp_get_wtime();
  printf("Running Random SGD with SVD init.. \n");
  p.run_sgd_random(100., 1e-2, 1e-5, INIT_SVD);
  printf("%d threads, randSGD takes %f seconds until error %f \n", n_threads, omp_get_wtime() - time, p.compute_testerror());

  time = omp_get_wtime();
  printf("Running NOMADi SGD.. \n");
  p.run_sgd_nomad_item(100., 1e-1, 1e-5, INIT_RANDOM);
  printf("%d threads, nomadSGDi takes %f seconds, error %f \n", n_threads, omp_get_wtime() - time, p.compute_testerror());

  time = omp_get_wtime();
  printf("Running NOMADu SGD.. \n");
  p.run_sgd_nomad_user(100., 1e-1, 1e-5, INIT_RANDOM);
  printf("%d threads, nomadSGDu takes %f seconds, error %f \n", n_threads, omp_get_wtime() - time, p.compute_testerror());
*/
  return 0;
}
