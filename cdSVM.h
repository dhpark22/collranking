/*Parallal Collaborative Ranking, collaborative project by Jin Zhang and Dohyung Park
Input: 
	**x,	2D arrary of feature_node, index :  used by w, 
	l,	row number of X, or number of training samples
	n,	column number, or dimension of each training samples
	C,	coefficient
Output:
	w,	feature vector, with dimension n

using dual coordinate descent with the equation listed in the report
*/

#include <algorithm>	// random shuffle
#include <stdlib.h>	// max

// learning U, with global w and alpha vector
// w is the U[iusr ... iusr+rank]
void trainU2(problem* p, parameter* param, double* U, int iusr, double* alpha) {
	//vector<int> order(p->l);
	double one_2C = 0.5 / param->C;
	double maxiter = 10;
	double eps = param->eps;
	double oldQ = 0;

	//for (int i = 0; i < p->l; ++i) {
	//	order[i] = i;
	//}

	for (int iter = 0; iter < maxiter; ++iter) {
		//random_shuffle(order.begin(), order.end() );
		for (int i = 0; i < p->l; ++i) {
			int idx = i;
			feature_node* xi = p->x[idx];

			double xi_snorm = 0;
			for (int j = 0; j < p->n; ++j) {
				xi_snorm += xi[j].value * xi[j].value;
			}

			double ywxi = 0;
			for (int j = 0; j < p->n; ++j) {
				ywxi += xi[j].value * U[iusr * p->n + j];
			}
			ywxi *= p->y[idx];

			double delta = (1 - ywxi - alpha[idx] * one_2C) / (xi_snorm + one_2C);
			delta = max(0., delta + alpha[idx]) - alpha[idx];
			alpha[idx] += delta;
			
			for (int j = 0; j < p->n; ++j) {
				double dval = delta * xi[j].value;
				U[idx * p->n + j] += dval;
			}
		}
		
		double w_snorm = 0;
		double alpha_snorm = 0;
		double alpha_sum = 0;
		double Q = 0;
		for (int j = 0; j < p->n; ++j) {
			w_snorm += U[iusr * p->n + j] * U[iusr * p->n + j];
		}
		for (int j = 0; j < p->l; ++j) {
			alpha_snorm += alpha[j] * alpha[j];	
			alpha_sum += alpha[j];
		} 
		Q = w_snorm / 2 + alpha_snorm * one_2C / 2 - alpha_sum;
		if (fabs(Q - oldQ) < eps) {
			break;
		}
		oldQ = Q;
	}
}

// learning V with global w and alpha vector 
// w is V[iid1 .. iid1+rank], -V[iid2 .. iid2+rank]
void trainV2(problem* p, parameter* param, double* V, comparison& c, double& alpha ) {
	double one_2C = 0.5 / param->C;
	double maxiter = 1000;
	double eps = param->eps;
	double oldQ = 0;  
	feature_node* xi = p->x[0];

	int uid = c.user_id;
	int iid1 = c.item1_id;
	int iid2 = c.item2_id;
	int rank = p->n / 2;

	for (int iter = 0; iter < maxiter; ++iter) {
		double xi_snorm = 0;
		for (int j = 0; j < rank; ++j) {
			xi_snorm += xi[j].value * xi[j].value;
		}
		xi_snorm *= 2;

		double ywxi = 0;
		for (int j = 0; j < rank; ++j) {
			ywxi += xi[j].value * V[iid1 * rank + j];
			ywxi += xi[j + rank].value * V[iid2 * rank + j];
		}

		double delta = (1 - ywxi - alpha * one_2C) / (xi_snorm * 2 + one_2C);		// xi_snorm * 2, this is the only difference

		if (fabs(delta) < eps || alpha == 0 && delta < 0) {
			break;
		}
		
		delta = max(0., delta + alpha) - alpha;
		alpha += delta;
		for (int j = 0; j < rank; ++j) {
			double dval = delta * xi[j].value;
			//#pragma omp atomic
			V[iid1 * rank + j] += dval;
			//#pragma omp atomic
			V[iid2 * rank + j] -= dval;
		}

	}
}
