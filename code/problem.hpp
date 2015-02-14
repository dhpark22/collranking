#ifndef __PROBLEM_HPP__
#define __PROBLEM_HPP__

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <algorithm>
#include <vector>
#include <fstream>

#include "elements.hpp"
#include "loss.hpp"

using namespace std;

class Problem {
  public: 
    int n_users, n_items, n_train_comps; // number of users/items in training sample, number of samples in traing and testing data set
    double lambda;

    loss_option_t loss_option = L2_HINGE;

    vector<comparison>   train;
    vector<int>          tridx;

    Problem();
    Problem(loss_option_t, double);				// default constructor
    ~Problem();					// default destructor
    void read_data(const std::string&);	// read function
  
    int get_nusers() { return n_users; }
    int get_nitems() { return n_items; }
    double evaluate(Model& model);
};

// may be more parameters can be specified here
Problem::Problem() {
}

Problem::Problem (loss_option_t option, double l) : lambda(l), loss_option(option) { 
}

Problem::~Problem () {
}

void Problem::read_data(const std::string &train_file) {

  // Prepare to read files
  n_users = n_items = 0;
  ifstream f;

  // Read training comparisons
  f.open(train_file);
  if (f.is_open()) {
    int uid, i1id, i2id, uid_current = 0;
    tridx.resize(0);
    tridx.push_back(0);
    while (f >> uid >> i1id >> i2id) {
      n_users = max(uid, n_users);
      n_items = max(i1id, max(i2id, n_items));
      --uid; --i1id; --i2id; // now user_id and item_id starts from 0

      while(uid > uid_current) {
        std::sort(train.begin()+tridx[uid_current], train.end(), comp_userwise);
        tridx.push_back(train.size());
        ++uid_current;
      }

      train.push_back(comparison(uid, i1id, i2id, 1));
    }

    std::sort(train.begin()+tridx[uid_current], train.end(), comp_userwise);
    tridx.push_back(train.size());
   
    n_train_comps = train.size();

  } else {
    printf("Error in opening the training file!\n");
    exit(EXIT_FAILURE);
  }
  f.close();

  printf("%d users, %d items, %d comparisons\n", n_users, n_items, n_train_comps);

}	

double Problem::evaluate(Model& model) {
  double l = compute_loss(model, train, loss_option);
  double u = model.Unormsq();
  double v = model.Vnormsq();
 
  double f = l + .5*lambda*(u+v);

  printf("%f, %f, %f, %f", f, l, u, v);

  return f;
}


#endif
