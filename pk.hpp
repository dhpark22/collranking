#ifndef __PK_HPP__
#define __PK_HPP__

#include <utility>
#include <vector>
#include <queue>
#include <iostream>
#include <fstream>
#include <unordered_set>

#include "model.hpp"
#include "ratings.hpp"
#include "loss.hpp"

class Evaluator {
  
  Evaluator();
  virtual void evaluate(const Model&);

};

class EvaluatorBinary : public Evaluator {
  std::vector<std::unordered_set<int> > train, test;	
  std::vector<int> k;
  // please add your structure to store the datasets

  public:
<<<<<<< HEAD
    void load_files();
};
=======
    void load_files(char*, char*, std::vector<int>&);
}
>>>>>>> 1b481e9e5095257ad0bf4fed42b9256032017247

class EvaluatorRating : public Evaluator {

  RatingMatrix test;

  public:
    void load_files();  
};

void EvaluateRating::load_files (char* test_ratings) {

  test.read_lsvm(test_ratings);
  test.compute_dcgmax(10);

}

void EvaluateRating::evaluate(const Model& model) {
  
  pair<double,double> err = compute_pairwiseError(test, model);
  double ndcg = compute_ndcg(test, model);

  printf(" %f %f %f ", err.first, err.second, ndcg);

}

struct pkcomp {
	bool operator() (std::pair<int, double> i, std::pair<int, double> j) {
		return i.second > j.second;
	}
};

void EvaluateBinary::load_files (char* train_repo, char* test_repo, std::vector<int>& ik) {
	std::ifstream tr(train_repo);
	if (tr) {
		int uid, iid;
		while (tr >> uid >> iid) {
			train[uid - 1].insert(iid - 1);
		}
	} else {
		printf ("Error in opening the training repository!\n");
		exit(EXIT_FAILURE);
	}
	tr.close();

	std::ifstream te(test_repo);
	if (te) {
		int uid, iid;
		while (te >> uid >> iid) {
			test[uid - 1].insert(iid - 1);
		}
	} else {
		printf ("Error in opening the testing repository!\n");
		exit(EXIT_FAILURE);
	}
	te.close();

	k = ik;
} 

void EvaluateBinary::evaluate (const Model& model) {
	train.resize(model.n_users);
	test.resize(model.n_users);
	
	int maxK = k[k.size() - 1];
	std::vector<double> ret(k.size(), 0);
	std::priority_queue<std::pair<int, double>, std::vector<std::pair<int, double> >, pkcomp> pq;	

	for (int i = 0; i < model.n_users; ++i) {
		for (int j = 0; j < model.n_items; ++j) {
			if (train[i].find(j) == train[i].end() ) {
				continue;
			}

			double score = 0;
			double *user_vec = &model.U[i * model.rank];
			double *item_vec = &model.V[j * model.rank];
			for (int l = 0; l < model.rank; ++l) {
				score += user_vec[l] * item_vec[l];
			}

			if (pq.size() < maxK) {
				pq.push(std::pair<int, double>(j, score) );
			} else if (pq.top().second < score) {
				pq.push(std::pair<int, double>(j, score) );
				pq.pop();	
			}
		}

		int ps = pq.size();
		while (ps) {
			int item = pq.top().first;
			for (int j = k.size() - 1; j >= 0; --j) {
				if (ps > k[j]) break;
				if (test[i].find(item) != test[i].end() ) ++ret[j];
			}
			pq.pop();
			--ps;
		}
	}

	for (int i = 0; i < k.size(); ++i) {
		ret[i] = ret[i] / model.n_users / k[i];
	}

	printf("compute precision at k\n");
	for (int i = 0; i < k.size(); ++i) {
		printf("k %d, precision %f\n", k[i], ret[i]);
	}
}

#endif
