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
// Compile: g++ -std=C++11 -O3 -g -fopenmp collaborative_ranking.cpp
// Run: ./a.out [rating_file] [rating_format] [graph_output] [num_partitions]

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
	vector<comparison> comparisons_test;			// vector stores testing comparison data

	void de_allocate();					// deallocate U, V when they are used multiple times by different methods
	public:
		Problem(int, int);				// default constructor
		~Problem();					// default destructor
	void read_data(char* train_file, char* test_file);	// read function
	void alt_rankSVM();
	// two different sgd function
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

	this->U = new double [this->n_users * this->rank];
	this->V = new double [this->n_items * this->rank];

	if (!is_clustered) {
		this->g.cluster();		// call graph clustering prior to the computation
		is_clustered = true;
	}
}

void Problem::alt_rankSVM () {
	srand(time(NULL));
	for (int i = 0; i < this->n_users * this->rank; ++i) {
		this->U[i] = ((double) rand() / RAND_MAX);
	}
	for (int i = 0; i < this->n_items * this->rank; ++i) {
		this->V[i] = ((double) rand() / RAND_MAX);
	}

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
		for (int j = 0; j < 2 * this->rank; ++j) {
			B[i][j].index = j + 1;
		}
		B[i][this->rank * 2].index = -1;
	}

	for (int iter = 0; iter < 20; ++iter) {
		// Learning U
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
			double *y = new double[P.l];
			for (int j = 0; j < P.l; ++j) {
				y[j] = 1.;
			}
			P.y = y;
			P.x = &A[this->g.uidx[i]];
			P.bias = -1.;

			struct parameter param;
			param.solver_type = L2R_L2LOSS_SVC_DUAL;
			param.C = 1.;
			param.eps = 1e-8;
			struct model *M;
			if (!check_parameter(&P, &param) ) {
				// run SVM
				M = train(&P, &param);
				//vector<double> w = trainU(&P, &param);
				// store the result
				for (int j = 0; j < rank; ++j) {
					this->U[i * rank + j] = M->w[j];
				}
				free_and_destroy_model(&M);
			}
			delete [] y;
		}

		// Learning V 
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
				double y = 1.;
				P.y = &y;
				P.x = &B[j];
				P.bias = -1;

				struct parameter param;
				param.solver_type = L2R_L2LOSS_SVC_DUAL;
				param.C = 1.;
				param.eps = 1e-8;
				struct model *M;
				if (!check_parameter(&P, &param) ) {
					// run SVM
					M = train(&P, &param);
					// vector<double> w = trainV(&P, &param);

					// store the result
					for (int s = 0; s < rank; ++s) {
						int v1 = this->g.pcmp[j].item1_id;
						int v2 = this->g.pcmp[j].item2_id;
						this->V[this->g.pcmp[j].item1_id * this->rank + s] = M->w[s];			// other threads might be doing the same thing
						this->V[this->g.pcmp[j].item2_id * this->rank + s] = M->w[s + this->rank];		// so add lock to the two steps is another option.
					}
					free_and_destroy_model(&M);
				}
			}
		}
		printf("iteratrion %d, test error %f\n", iter, this->compute_testerror() );
	}

	for (int i = 0; i < this->n_train_comps; ++i) {
		delete [] A[i];
		delete [] B[i];
	}
	delete [] A;
	delete [] B;
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
	for(int i=0; i<n_test_comps; i++) {
		double prod = 0.;
		int user_idx  = comparisons_test[i].user_id * rank;
		int item1_idx = comparisons_test[i].item1_id * rank;
		int item2_idx = comparisons_test[i].item2_id * rank;
		for(int k=0; k<rank; k++) prod += U[user_idx + k] * (V[item1_idx + k] - V[item2_idx + k]);
		if (prod < 0.) n_error++;
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
	//int nparts = (nr_threads > 1) ? nr_threads : 2;
	Problem p(10, nr_threads + 1);		// rank = 10, #partition = 16
	p.read_data(argv[1], argv[2]);
	omp_set_dynamic(0);
	omp_set_num_threads(nr_threads);
	double start = omp_get_wtime();
	p.alt_rankSVM();
	double end = omp_get_wtime() - start;
	printf("%d threads, takes %f seconds\n", nr_threads, end);
	return 0;
}
