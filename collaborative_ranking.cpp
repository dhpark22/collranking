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
// Compile: g++ -std=c++11 -O3 -g -fopenmp collaborative_ranking.cpp
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
#include "collaborative_ranking.h"
#include "liblinear-1.94/linear.h"

using namespace std;

class Problem {
	bool is_allocated, is_clustered;
	int n_users, n_items, n_train_comps, n_test_comps; 	// number of users/items in training sample, number of samples in traing and testing data set
	int rank, lambda, nparts;				// parameters
	double *U, *V;						// low rank U, V
	double alpha, beta;					// parameter for sgd
	Graph g;						// Graph used for clustering training data
	vector<int> n_comps_by_user, n_comps_by_item;	// number of comparisons for each user and item
	vector<comparison> comparisons_test;			// vector stores testing comparison data

	void de_allocate();					// check if U, V has been allocated
	bool sgd_step(const int &idx, const double step_size);

	public:
		Problem(int, int);					// default constructor
	void read_data(char* train_file, char* test_file);	// read function
	void alt_rankSVM();
	void run_sgd_random();
	void run_sgd_nomad();
	// two different sgd function
	double compute_ndcg();
	double compute_testerror();
};

// may be more parameters can be specified here
Problem::Problem (int r, int np): g(np) {
	this->rank = r;
	this->is_allocated = false;
	this->nparts = np;
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
		printf("Error in opening the testing file\n");
		exit(EXIT_FAILURE);
	}
	f.close();
}

void Problem::alt_rankSVM () {
	if (!is_clustered) {
		this->g.cluster();		// call graph clustering prior to the computation
	}

	if (this->is_allocated) {
		this->de_allocate();
	}
	this->is_allocated = true;

	this->U = new double (this->n_users * this->rank);
	this->V = new double (this->n_items * this->rank);

	srand(time(NULL));
	for (int i = 0; i < this->n_users; ++i) {
		U[i] = ((double) rand() / RAND_MAX);
	}
	for (int i = 0; i < this->n_users; ++i) {
		V[i] = ((double) rand() / RAND_MAX);
	}

	// Alternating RankSVM
	struct feature_node **A, **B;
	A = new struct feature_node*[this->n_train_comps];
	for (int i = 0; i < this->n_train_comps; ++i) {
		A[i] = new struct feature_node[this->rank + 1];
		for (int j = 0; j < this->rank; j++) {
			A[i][j].index = j;
		}
		A[i][this->rank].index = -1;
	}

	B = new struct feature_node*[this->n_train_comps];
	for (int i = 0; i < this->n_train_comps; ++i) {
		B[i] = new struct feature_node[this->rank * 2 + 1];
		for (int j = 0; j < 2 * this->rank; ++j) {
			B[i][j].index = j;
		}
		B[i][this->rank * 2].index = -1;
	}

	for (int iter = 0; iter < 20; ++iter) {
		// Learning U
		// #pragma omp parallel for
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
			struct model *M;
			if (!check_parameter(&P, &param) ) {
				// run SVM
				M = train(&P, &param);
				// store the result
				for (int j = 0; j < rank; ++j) {
					U[i * rank + j] = M->w[j];
				}
				//free_and_destory_model(&M);
			}
		}

		// Learning V 
		// #pragma omp parallel for
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

				struct model *M;
				if (!check_parameter(&P, &param) ) {
					// run SVM
					M = train(&P, &param);

					// store the result
					for (int s = 0; s < rank; ++s) {
						V[this->g.pcmp[j].item1_id * this->rank + s] = M->w[s];			// other threads might be doing the same thing
						V[this->g.pcmp[j].item2_id * this->rank + s] = M->w[s + this->rank];		// so add lock to the two steps is another option.
					}
					//free_and_destroy_model(&M);
				}
			}
		}
	}
}

bool Problem::sgd_step(const int &idx, const double step_size) {
	double *user_vec  = &U[g.ucmp[idx].user_id * rank];
	double *item1_vec = &V[g.ucmp[idx].item1_id * rank];
	double *item2_vec = &V[g.ucmp[idx].item2_id * rank];

    int n_comps_user  = n_comps_by_user[g.ucmp[idx].user_id];
    int n_comps_item1 = n_comps_by_item[g.ucmp[idx].item1_id];
    int n_comps_item2 = n_comps_by_item[g.ucmp[idx].item2_id];

	double err = 1.;
	for(int k=0; k<rank; k++) err -= user_vec[k] * (item1_vec[k] - item2_vec[k]);

	if (err > 0) {	
		double grad = -2 * err;		// gradient direction for l2 hinge loss

		for(int k=0; k<rank; k++) {
			double user_dir  = (grad * (item1_vec[k] - item2_vec[k]) + lambda / n_comps_user * user_vec[k]);
			double item1_dir = (grad * user_vec[k] + lambda / n_comps_item1 * item1_vec[k]);
			double item2_dir = (-grad * user_vec[k] + lambda / n_comps_item2 * item2_vec[k]);

			user_vec[k]  -= step_size * user_dir;
			item1_vec[k] -= step_size * item1_dir;
			item2_vec[k] -= step_size * item2_dir;
		}

		return true;
	}

	return false;
}

void Problem::run_sgd_random() {

	// Clustering
	/* Cluster the triples based on comparisons_train */

	srand(time(NULL));
	for(int i=0; i<n_users*rank; i++) U[i] = ((double)rand()/(RAND_MAX));
	for(int i=0; i<n_items*rank; i++) V[i] = ((double)rand()/(RAND_MAX));

    int n_iter = 1e4;
	for(int iter=1; iter<n_iter; iter++) {
		int idx = (int)((double)rand() * (double)n_train_comps / (double)RAND_MAX);

		sgd_step(idx, alpha / (1. + beta/(double)iter));

		double ndcg = compute_ndcg();
		double test_err = compute_testerror();
	}

}

void Problem::run_sgd_nomad() {

	// Clustering
	/* Cluster the triples based on comparisons_train */

	srand(time(NULL));
	for(int i=0; i<n_users*rank; i++) U[i] = ((double)rand()/(RAND_MAX));
	for(int i=0; i<n_items*rank; i++) V[i] = ((double)rand()/(RAND_MAX));

    int n_iter = 1e4;
	for(int iter=1; iter<n_iter; iter++) {
		int idx = (int)((double)rand() * (double)n_train_comps / (double)RAND_MAX);

		sgd_step(idx, alpha / (1. + beta/(double)iter));

		double ndcg = compute_ndcg();
		double test_err = compute_testerror();
	}

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
	delete this->U;
	delete this->V;
}

int main (int argc, char* argv[]) {
	Problem p(10, 16);		// rank = 10, #partition = 16
	p.read_data(argv[1], argv[2]);
	// p.alt_rankSVM();
	// p.run_sgd_random();
	return 0;
}
