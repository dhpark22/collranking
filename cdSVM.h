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

// learning U, return w vector
vector<double> trainU(problem* p, parameter* param) {
	vector<int> order(p->l);
	vector<double> alpha(p->l, 0);
	vector<double> w(p->n, 0);
	double one_2C = 0.5 / param->C;
	double maxiter = 1000;
	double eps = param->eps;
	double oldQ = 0;

	for (int i = 0; i < p->l; ++i) {
		order[i] = i;
	}

	for (int iter = 0; iter < maxiter; ++iter) {
		random_shuffle(order.begin(), order.end() );
		for (int i = 0; i < p->l; ++i) {
			int idx = order[i];
			feature_node* xi = p->x[idx];

			double xi_snorm = 0;
			for (int j = 0; j < p->n; ++j) {
				xi_snorm += xi[j].value * xi[j].value;
			}

			double ywxi = 0;
			for (int j = 0; j < p->n; ++j) {
				ywxi += xi[j].value * w[xi[j].index - 1];
			}
			ywxi *= p->y[idx];

			double delta = (1 - ywxi - alpha[idx] * one_2C) / (xi_snorm + one_2C);
			delta = max(0., delta + alpha[idx]) - alpha[idx];
			alpha[idx] += delta;
			
			for (int j = 0; j < p->n; ++j) {
				w[xi[j].index - 1] += delta * p->y[idx] * xi[j].value;
			}
		}
		
		double w_snorm = 0;
		double alpha_snorm = 0;
		double alpha_sum = 0;
		double Q = 0;
		for (int j = 0; j < p->n; ++j) {
			w_snorm += w[j] * w[j];
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
	return w;
}

// learning V, return w vector
// the input is only one sample, or p->n = 1
vector<double> trainV(problem* p, parameter* param) {
	double alpha = 0;
	double one_2C = 0.5 / param->C;
	double maxiter = 1000;
	double eps = param->eps;
	double oldQ = 0;  
	vector<double> w(p->n);
	feature_node* xi = p->x[0];

	for (int iter = 0; iter < maxiter; ++iter) {
		double xi_snorm = 0;
		for (int j = 0; j < p->n; ++j) {
			xi_snorm += xi[j].value * xi[j].value;
		}

		double ywxi = 0;
		for (int j = 0; j < p->n; ++j) {
			ywxi += xi[j].value * w[xi[j].index - 1];
		}

		double delta = (1 - ywxi - alpha * one_2C) / (xi_snorm * 2 + one_2C);		// xi_snorm * 2, this is the only difference
		delta = max(0., delta + alpha) - alpha;
		alpha += delta;

		for (int j = 0; j < p->n; ++j) {
			w[xi[j].index - 1] += delta * p->y[0] * xi[j].value;
		}

		// calculating the objective function value
		double Q = 0;
		for (int j = 0; j < p->n; ++j) {
			Q += w[j] * w[j];
		}
		Q = Q / 2 + alpha * alpha * one_2C / 2 - alpha;
		
		if (fabs(Q - oldQ) < eps) {
			break;
		}
		oldQ = Q;
	}
	return w;
}
