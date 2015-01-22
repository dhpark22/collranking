#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <iostream>
#include <string>
#include <time.h>
#include <math.h>
#include <omp.h>
#include <vector>
#include <algorithm>
#include <Eigen/Dense>
#include <Eigen/Core>
#define EIGEN_DONT_PARALLELIZE

#include "model.hpp"
#include "ratings.hpp"
#include "eval.hpp"

void als(Model &model, const RatingMatrix &train, const RatingMatrix &test, int nIter, double lambda) {

  int n_users = model.n_users;
  int n_items = model.n_items;
  int rank    = model.rank;

  int n_ratings = train.ratings.size();

	double **Usub = new double*[n_ratings];
	double **Vsub = new double*[n_ratings];
	
  std::vector<std::pair<int,int> > item_idx(n_ratings);
  std::vector<int> iidx(n_items+1);

  for(int i=0; i<n_ratings; i++) {
    Usub[i] = &(model.U[train.ratings[i].user_id * rank]);
		Vsub[i] = &(model.V[train.ratings[i].item_id * rank]);

    item_idx[i].first = train.ratings[i].item_id;
    item_idx[i].second = i; 
  }
  
  std::sort(item_idx.begin(), item_idx.end());

  iidx[0] = 0; iidx[n_items] = n_ratings;
  int i = 0;
  for(int iid=0; iid<n_items; ++iid) {
    while((i < n_ratings) && (item_idx[i].first < iid)) ++i;
    iidx[iid] = i;
  }

  int n_max = 0;
  for(int uid=0; uid<n_users; ++uid) n_max = std::max(n_max,train.idx[uid+1]-train.idx[uid]);
  for(int iid=0; iid<n_items; ++iid) n_max = std::max(n_max,iidx[iid+1]-iidx[iid]);

	// random initialization 
	srand(time(NULL));
	for(int i=0; i<n_users*rank; i++) model.U[i] = ((double)rand()/(double)RAND_MAX); 
  for(int i=0; i<n_items*rank; i++) model.V[i] = ((double)rand()/(double)RAND_MAX); 

  Eigen::setNbThreads(1);
  Eigen::MatrixXd A(rank,rank);
  Eigen::VectorXd b(rank);
  
  double *Xsrc = new double[rank * n_max];
  double *ysrc = new double[n_max]; 

  std::pair<double,double> perror = compute_pairwiseError(test, model);
  double ndcg = compute_ndcg(test, model);
  printf("0: %f %f %f\n", perror.first, perror.second, ndcg);

  int n;
  for(int iter=1; iter<=nIter; ++iter) {

		// ridge regression for U
    for(int uid=0; uid<n_users; ++uid) {
      n = train.idx[uid+1] - train.idx[uid];

			if (n > 0) {
        for(int j=0; j<n; j++) ysrc[j] = train.ratings[train.idx[uid]+j].score;
        for(int j=0; j<n; j++) memcpy(Xsrc+rank*j, Vsub[train.idx[uid]+j], sizeof(double)*rank);

        Eigen::Map<Eigen::MatrixXd> X(Xsrc,rank,n);
        Eigen::Map<Eigen::VectorXd> y(ysrc,n);
        Eigen::Map<Eigen::VectorXd> w(&(model.U[uid*rank]),rank); 
			  
        A = (lambda * Eigen::MatrixXd::Identity(rank,rank) + X * X.transpose());
				b = X * y;
				w = A.llt().solve(b);

			}
    }
    
		// ridge regression for M
    for(int iid=0; iid<n_items; ++iid) {
      n = iidx[iid+1] - iidx[iid];	

      if (n > 0) {
			  for(int j=0; j<n; j++) ysrc[j] = train.ratings[item_idx[iidx[iid]+j].second].score;
        for(int j=0; j<n; j++) memcpy(Xsrc+rank*j, Usub[item_idx[iidx[iid]+j].second], sizeof(double)*rank);

        Eigen::Map<Eigen::MatrixXd> X(Xsrc,rank,n);
        Eigen::Map<Eigen::VectorXd> y(ysrc,n);
        Eigen::Map<Eigen::VectorXd> w(&(model.V[iid*rank]),rank); 
	
			  A = (lambda * Eigen::MatrixXd::Identity(rank,rank) + X * X.transpose());
				b = X * y;
				w = A.llt().solve(b);

      }
    }

    std::pair<double,double> perror = compute_pairwiseError(test, model);
    double ndcg = compute_ndcg(test, model);
    printf("%d: %f %f %f\n", iter, perror.first, perror.second, ndcg);

  }	

  delete[] Xsrc;
  delete[] ysrc;
}


int main(int argc, char *argv[]) {

  RatingMatrix train, test;

  std::string train_filename(argv[1]), test_filename(argv[2]);
  train.read_lsvm(train_filename);
  test.read_lsvm(test_filename);
  test.compute_dcgmax(10);

  int n_users = std::max(train.n_users, test.n_users);
  int n_items = std::max(train.n_items, test.n_items);
  
  Model model(n_users, n_items, 10);

 	double lambda = atof(argv[3]);
	int nr_threads = atoi(argv[4]);

  als(model, train, test, 30, lambda);

}


