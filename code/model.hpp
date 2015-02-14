#ifndef __MODEL_HPP__
#define __MODEL_HPP__

class Model {
  public:
    bool is_allocated;
    int n_users, n_items;           // number of users/items in training sample, number of samples in traing and testing data set
    int rank;                       // parameters
    double *U, *V;                  // low rank U, V

    void allocate(int nu, int ni);    
    void de_allocate();					    // deallocate U, V when they are used multiple times by different methods

    Model(int r): is_allocated(false), rank(r) {}
    Model(int nu, int ni, int r): is_allocated(false), rank(r) { allocate(nu, ni); }
 
    double Unormsq();
    double Vnormsq();
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
