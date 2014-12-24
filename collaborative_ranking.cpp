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
#include "linear.h"
#include "collaborative_ranking.h"

using namespace std;

class Problem {
    bool is_allocated, is_clustered;
    int n_users, n_items, n_train_comps, n_test_comps; 	// number of users/items in training sample, number of samples in traing and testing data set
    int rank, lambda, nparts;				// parameters
    double *U, *V;						// low rank U, V
    double alpha, beta;					// parameter for sgd
    Graph g;						// Graph used for clustering training data
    vector<int> n_comps_by_user, n_comps_by_item;
    vector<comparison> comparisons_test;			// vector stores testing comparison data

    bool sgd_step(const comparison &comp, const double l, const double step_size);
    void de_allocate();					// deallocate U, V when they are used multiple times by different methods
  
  public:
    Problem(int, int);				// default constructor
    ~Problem();					// default destructor
    void read_data(char* train_file, char* test_file);	// read function
    void alt_rankSVM();
    void run_sgd_random();
    void run_sgd_nomad();
    double compute_ndcg();
    double compute_testerror();
};

// may be more parameters can be specified here
Problem::Problem (int r, int np): g(np) {
	this->rank = r;
	this->is_allocated = false;
	this->is_clustered = false;
	this->nparts = np;
}

Problem::~Problem () {
	//printf("calling the destructor\n");
	this->de_allocate();
}

void Problem::read_data (char* train_file, char* test_file) {
	this->g.read_data(train_file);	// training file will be feed to graph for clustering
	this->n_users = this->g.n;
	this->n_items = this->g.m;
	this->n_train_comps = this->g.omega;

	n_comps_by_user.clear(); n_comps_by_user.resize(this->n_users);
	n_comps_by_item.clear(); n_comps_by_item.resize(this->n_items);
	for(int i=0; i<this->n_users; i++) n_comps_by_user[i] = 0;
	for(int i=0; i<this->n_items; i++) n_comps_by_item[i] = 0;

	for(int i=0; i<this->n_train_comps; i++) {
		++n_comps_by_user[g.ucmp[i].user_id];
		++n_comps_by_item[g.ucmp[i].item1_id];
		++n_comps_by_item[g.ucmp[i].item2_id];
	}

	ifstream f(test_file);
	if (f) {
		int u, i, j;
		while (f >> u >> i >> j) {
			this->n_users = max(u, this->n_users);
			this->n_items = max(i, max(j, this->n_items));
			this->comparisons_test.push_back(comparison(u - 1, i - 1, j - 1) );
		}
		this->n_test_comps = this->comparisons_test.size();
	} else {
		printf("error in opening the testing file\n");
		exit(EXIT_FAILURE);
	}
	f.close();

    printf("%d users, %d items, %d training comps, %d test comps \n", this->n_users, this->n_items,
                                                                      this->n_train_comps,
                                                                      this->n_test_comps);

	this->U = new double [this->n_users * this->rank];
	this->V = new double [this->n_items * this->rank];

}	

void Problem::alt_rankSVM () {
	if (!is_clustered) {
		this->g.cluster();		// call graph clustering prior to the computation
		is_clustered = true;
	}

	double eps = 1e-8;
	srand(time(NULL));
	for (int i = 0; i < this->n_users * this->rank; ++i) {
		this->U[i] = ((double) rand() / RAND_MAX);
	}
	for (int i = 0; i < this->n_items * this->rank; ++i) {
		this->V[i] = ((double) rand() / RAND_MAX);
	}

	double *alphaV = new double[this->n_train_comps];
	double *alphaU = new double[this->n_train_comps];
	memset(alphaV, 0, sizeof(double) * this->n_train_comps);
	memset(alphaU, 0, sizeof(double) * this->n_train_comps);

	printf("initial error %f\n", this->compute_testerror() );

	// Alternating RankSVM
	struct feature_node **A, **B;
	A = new struct feature_node*[this->n_train_comps];
	for (int i = 0; i < this->n_train_comps; ++i) {
		A[i] = new struct feature_node[this->rank + 1];
		for (int j = 0; j < this->rank; ++j) {
			A[i][j].index = j + 1;
		}
		A[i][this->rank].index = -1;
	}

	B = new struct feature_node*[this->n_train_comps];
	for (int i = 0; i < this->n_train_comps; ++i) {
		B[i] = new struct feature_node[this->rank * 2 + 1];
		for (int j = 0; j < this->rank; ++j) {
			B[i][j].index = this->g.pcmp[i].item1_id * rank + j + 1;
			B[i][j + rank].index = this->g.pcmp[i].item2_id * rank + j + 1;
		}
		B[i][this->rank * 2].index = -1;
	}

    printf("Initial test error : %f \n", this->compute_testerror());

	for (int OuterIter = 0; OuterIter < 20; ++OuterIter) {
		// Learning U
		double start = omp_get_wtime();
		#pragma omp parallel for
		for (int i = 0; i < this->n_users; ++i) {
			for (int j = this->g.uidx[i]; j < this->g.uidx[i + 1]; ++j) {
				double *V1 = &V[this->g.ucmp[j].item1_id * this->rank];
				double *V2 = &V[this->g.ucmp[j].item2_id * this->rank];
				for (int s = 0; s < this->rank; ++s) {
					A[j][s].value = V1[s] - V2[s];
				}
			}

			// call LIBLINEAR with U[i * rank]
			struct problem P;
			P.l = this->g.uidx[i + 1] - this->g.uidx[i];
			P.n = this->rank;
			P.x = &A[this->g.uidx[i]];
			P.bias = -1.;

			struct parameter param;
			param.solver_type = L2R_L2LOSS_SVC_DUAL;
			param.C = 1.;
			param.eps = eps;
			if (!check_parameter(&P, &param) ) {
				// run SVM
				trainU(&P, &param, U, i, &alphaU[this->g.uidx[i] ]);
			}
		}
		double Utime = omp_get_wtime() - start;
		printf("iteratrion %d, test error %f \n", OuterIter, this->compute_testerror());
	
        // Learning V 
		#pragma omp parallel for
		for (int i = 0; i < this->nparts; ++i) {
			// solve the SVM problem sequentially for each sample in the partition
			for (int j = this->g.pidx[i]; j < this->g.pidx[i + 1]; ++j) {
				// generate the training set for V using U
				for (int s = 0; s < this->rank; ++s) {
					B[j][s].value = U[this->g.pcmp[j].user_id * this->rank + s];		// U_i
					B[j][s + rank].value = -U[this->g.pcmp[j].user_id * this->rank + s];	// -U_i
				}		
			
				// call LIBLINEAR with U[i*rank], B[j]
				struct problem P;
				P.l = 1;
				P.n = rank * 2;
				P.x = &B[j];
				P.bias = -1;

				struct parameter param;
				param.solver_type = L2R_L2LOSS_SVC_DUAL;
				param.C = 1.;
				param.eps = eps;
				if (!check_parameter(&P, &param) ) {
					// run SVM
					trainV(&P, &param, V, this->g.pcmp[j], &alphaV[j]);
				}
			}
		}
		double Vtime = omp_get_wtime() - Utime - start;
		printf("iteratrion %d, test error %f, learning U takes %f seconds, learning V takes %f seconds\n", OuterIter, this->compute_testerror(), Utime, Vtime);
	}

	for (int i = 0; i < this->n_train_comps; ++i) {
		delete [] A[i];
		delete [] B[i];
	}
	delete [] A;
	delete [] B;
	delete [] alphaV;
	delete [] alphaU;
}	

bool Problem::sgd_step(const comparison& comp, const double l, const double step_size) {
	double *user_vec  = &U[comp.user_id  * rank];
	double *item1_vec = &V[comp.item1_id * rank];
	double *item2_vec = &V[comp.item2_id * rank];

    int n_comps_user  = n_comps_by_user[comp.user_id];
    int n_comps_item1 = n_comps_by_item[comp.item1_id];
    int n_comps_item2 = n_comps_by_item[comp.item2_id];

    if ((n_comps_user < 1) || (n_comps_item1 < 1) || (n_comps_item2 < 1))
        printf("ERROR\n");

	double err = 1.;
	for(int k=0; k<rank; k++) err -= user_vec[k] * (item1_vec[k] - item2_vec[k]);

    /*
    if (err != err) 
    {
        printf("%d %d %d \n", comp.user_id, comp.item1_id, comp.item2_id);
        for(int k=0; k<rank; k++)
            printf("%f %f %f \n", user_vec[k], item1_vec[k], item2_vec[k]);
        printf("\n");
    }
    */

	if (err > 0) {	
		double grad = -2. * err;		// gradient direction for l2 hinge loss

		for(int k=0; k<rank; k++) {
			double user_dir  = step_size * (grad * (item1_vec[k] - item2_vec[k]) + l / (double)n_comps_user * user_vec[k]);
			double item1_dir = step_size * (grad * user_vec[k] + l / (double)n_comps_item1 * item1_vec[k]);
			double item2_dir = step_size * (-grad * user_vec[k] + l / (double)n_comps_item2 * item2_vec[k]);

            /*
            if ((user_dir != user_dir) || (item1_dir != item1_dir) || (item2_dir != item2_dir))
                printf("%f %f %f %f %f %d %d %d \n", grad, user_vec[k], item1_vec[k], item2_vec[k], l,
                                         n_comps_user, n_comps_item1, n_comps_item2);
            */

            //#pragma omp atomic
			user_vec[k]  -= user_dir;

            //#pragma omp atomic
			item1_vec[k] -= item1_dir;

            //#pragma omp atomic
			item2_vec[k] -= item2_dir;
		}

		return true;
	}

	return false;
}

void Problem::run_sgd_random() {

    auto real_rand = std::bind(std::uniform_real_distribution<double>(0,1), std::mt19937(time(NULL)));
	for(int i=0; i<n_users*rank; i++) U[i] = real_rand();
	for(int i=0; i<n_items*rank; i++) V[i] = real_rand();

    alpha = .1;
    beta  = 1e-5;
    lambda = 1;

    int n_threads = g.nparts;
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
                sgd_step(g.ucmp[idx], lambda, 
                         alpha/(1.+beta*pow((double)((n_updates+n_max_updates*icycle)*n_threads),1.)));
            }
        }

        double error = this->compute_testerror();
        if (error < 0.) break; 
        printf("%d: iter %d, error %f, time %f \n", n_threads, (icycle+1)*n_max_updates, error, omp_get_wtime() - time);
    }

}

void Problem::run_sgd_nomad() {

	if (!is_clustered) {
		this->g.cluster();		// call graph clustering prior to the computation
		is_clustered = true;
	}

    auto real_rand = std::bind(std::uniform_real_distribution<double>(0,1), std::mt19937(time(NULL)));
	for(int i=0; i<n_users*rank; i++) U[i] = real_rand();
	for(int i=0; i<n_items*rank; i++) V[i] = real_rand();

    alpha = .1;
    beta  = 1e-5;
    lambda = 1.;

    int n_threads = g.nparts;
    int n_max_updates = n_train_comps/10/n_threads;

    int **queue = new int*[n_threads];
    std::vector<int> front(n_threads), back(n_threads);
    for(int i=0; i<n_threads; ++i) queue[i] = new int[n_users+1];
   
    printf("Initial test error : %f \n", this->compute_testerror());

    for(int i=0; i<n_threads; ++i) {
        for(int j=(n_users*i/n_threads), k=0; j<(n_users*(i+1)/n_threads); ++j, ++k) queue[i][k] = j;
        front[i] = 0;
        back[i]  = (n_users*(i+1)/n_threads) - (n_users*i/n_threads);
    }

    std::vector<int> c(n_train_comps,0);

    int n_updates_total = 0;

    double time = omp_get_wtime();
    for(int icycle=0; icycle<20; ++icycle) {
/*
        for(int i=0; i<n_threads; ++i) {
            int idx = (i+icycle)%n_threads;
            for(int j=(n_users*i/n_threads), k=0; j<(n_users*(i+1)/n_threads); ++j, ++k) queue[idx][k] = j;
            front[idx] = 0;
            back[idx]  = (n_users*(i+1)/n_threads) - (n_users*i/n_threads);
        }
*/
 		int flag = -1;

        #pragma omp parallel shared(n_updates_total, flag, queue, front, back)
        {

        int tid = omp_get_thread_num();
        int tid_next = tid-1;
        if (tid_next < 0) tid_next = n_threads-1;

        int n_updates = 1;

//        printf("thread %d/%d beginning : users %d - %d  \n", tid, tid_next, queue[tid][front[tid]], queue[tid][back[tid]-1]);
        while((flag == -1) && (n_updates < n_max_updates)) {
            if (front[tid] != back[tid]) {
                
                int uid;
                
//                #pragma omp critical
                {
                uid = queue[tid][front[tid]];
                front[tid] = (front[tid]+1) % (n_users+1);
                }

                for(int idx=g.p2idx[tid][uid]; idx<g.p2idx[tid][uid+1]; ++idx) {
		            sgd_step(g.pcmp[idx], lambda, 
                             alpha/(1.+beta*(double)((n_updates+icycle*n_max_updates)*n_threads)));
                    ++n_updates;
                }

//                #pragma omp critical
                {
                queue[tid_next][back[tid_next]] = uid;
                back[tid_next] = (back[tid_next]+1) % (n_users+1);
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

        int i_thr = (flag+1) % n_threads;
        int idx = queue[i_thr][front[i_thr]];
        for(int i=flag+1; i<=flag+n_threads; ++i) {  
            i_thr = i % n_threads;
            int n_indices = (n_users*(i_thr+1)/n_threads) - (n_users*i_thr/n_threads);
           
            for(int j=0; j<n_indices; ++j) {
                queue[i_thr][j] = idx; 
                idx = (idx+1) % n_users;
            }     

            front[i_thr] = 0;
            back[i_thr]  = n_indices;
        }
/*
        int max=0;
        for(int i=0; i<n_train_comps; i++) if (max < c[i]) max=c[i];
        printf("max %d\n", max); 

        int min=max;
        for(int i=0; i<n_train_comps; i++) if (min > c[i]) min=c[i];
        printf("min %d\n", min);
*/
        //printf("%d %f \n", n_updates_total, omp_get_wtime() - time);

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
   
    /* 
    for(int i=0; i<100; i++) {
        int idx = (int)((double)n_test_comps * (double)rand() / (double)RAND_MAX);
		double prod = 0.;
		int user_idx  = comparisons_test[idx].user_id * rank;
		int item1_idx = comparisons_test[idx].item1_id * rank;
		int item2_idx = comparisons_test[idx].item2_id * rank;
		for(int k=0; k<rank; k++) prod += U[user_idx + k] * (V[item1_idx + k] - V[item2_idx + k]);
	    printf("%f ", i, prod);
    }
    printf("\n");
    */

    for(int i=0; i<n_users*rank; i++)
        if ((U[i] > 1e5) || (U[i] < -1e5)) {
            printf("large number : %d %d %f \n", i/rank, i%rank, U[i]);   
            return -1.;
        }

	for(int i=0; i<n_test_comps; i++) {
		double prod = 0.;
		int user_idx  = comparisons_test[i].user_id;
		int item1_idx = comparisons_test[i].item1_id;
		int item2_idx = comparisons_test[i].item2_id;
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

	int nr_threads = atoi(argv[3]);

	Problem p(10, nr_threads);		// rank = 10
	p.read_data(argv[1], argv[2]);
	omp_set_dynamic(0);
	omp_set_num_threads(nr_threads);
	double start = omp_get_wtime();
    
	//p.alt_rankSVM();
	//double m1 = omp_get_wtime() - start;
	//printf("%d threads, rankSVM takes %f seconds\n", nr_threads, m1);
	
    p.run_sgd_random();
	double m2 = omp_get_wtime() - start - m1;
    printf("%d threads, randSGD takes %f seconds\n", nr_threads, m2);

    p.run_sgd_nomad();
    double m3 = omp_get_wtime() - start - m2 - m1;
    printf("%d threads, nomadSGD takes %f seconds, error %f \n", nr_threads, m3, p.compute_testerror());
 
    return 0;
}
