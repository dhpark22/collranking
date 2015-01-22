#ifndef __MODEL_HPP__
#define __MODEL_HPP__

#include <algorithm>
#include "elements.hpp"

class Model {
  public:
    bool is_allocated;
    int n_users, n_items; 	// number of users/items in training sample, number of samples in traing and testing data set
    int rank;                       // parameters
    double *U, *V;                                  // low rank U, V

    void allocate(int nu, int ni);    
    void de_allocate();					            // deallocate U, V when they are used multiple times by different methods

    Model(int r): is_allocated(false), rank(r) {}
    Model(int nu, int ni, int r): is_allocated(false), rank(r) { allocate(nu, ni); }
 
    double Unormsq();
    double Vnormsq();
    double compute_loss(const std::vector<comparison>&, double);
    double compute_testerror(const std::vector<comparison>&);
};

double Model::Unormsq() {
  double p = 0.;
  for(int i=0; i<n_users*rank; ++i) p += U[i]*U[i];
  return p;
}

double Model::Vnormsq() {
  double p = 0.;
  for(int i=0; i<n_items*rank; ++i) p += V[i]*V[i];
  return p;
}

double Model::compute_loss(const std::vector<comparison>& TestComps, double lambda) {
  double p = 0., slack;
  for(int i=0; i<TestComps.size(); ++i) {
    double *user_vec  = &U[TestComps[i].user_id  * rank];
    double *item1_vec = &V[TestComps[i].item1_id * rank];
    double *item2_vec = &V[TestComps[i].item2_id * rank];
    double d = 0.;
    for(int j=0; j<rank; ++j) {
      d += user_vec[j] * (item1_vec[j] - item2_vec[j]);
    }
    slack = std::max(0., 1. - d);
    p += slack*slack/lambda;
  }
   
  return p;		
}

double Model::compute_testerror(const std::vector<comparison>& TestComps) {
	int n_error = 0; 
	
  for(int i=0; i<TestComps.size(); i++) {
		double prod = 0.;
		int user_idx  = TestComps[i].user_id;
		int item1_idx = TestComps[i].item1_id;
		int item2_idx = TestComps[i].item2_id;
		for(int k=0; k<rank; k++) prod += U[user_idx*rank + k] * (V[item1_idx*rank + k] - V[item2_idx*rank + k]);
		if (prod <= 0.) n_error += 1;
  }

  return (double)n_error / (double)(TestComps.size());
}

void Model::allocate(int nu, int ni) {
  if (is_allocated) de_allocate();

  U = new double[nu*rank];
  V = new double[ni*rank];

  n_users = nu;
  n_items = ni;

  is_allocated = true;
}

void Model::de_allocate () {
	if (!is_allocated) return;
  
  delete [] this->U;
	delete [] this->V;
	this->U = NULL;
	this->V = NULL;

  is_allocated = false;
}

#endif
