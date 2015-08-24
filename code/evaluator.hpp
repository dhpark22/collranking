#ifndef __EVALUATOR_HPP__
#define __EVALUATOR_HPP__

#include <utility>
#include <vector>
#include <algorithm>
#include <queue>
#include <iostream>
#include <fstream>
#include <unordered_set>

#include "model.hpp"
#include "ratings.hpp"
#include "loss.hpp"

class Evaluator {
  public: 
    virtual void evaluate(const Model&) {} 
    virtual void evaluateAUC(const Model&) {}
    virtual void load_files(const std::string&, const std::string&, std::vector<int>&) = 0;
 
    std::vector<int> k;
    int k_max;
};

class EvaluatorBinary : public Evaluator {
  public:
    std::vector<std::unordered_set<int> > train, test;	

    void load_files(const std::string&, const std::string&, std::vector<int>&);
    void evaluate(const Model&);
    void evaluateAUC(const Model&);
};

class EvaluatorRating : public Evaluator {
  RatingMatrix test;

  public:
    void load_files(const std::string&, const std::string&, std::vector<int>&);
    void evaluate(const Model&);
};

void EvaluatorRating::load_files (const std::string& train_repo, const std::string& test_repo, std::vector<int>& ik) {
  test.read_lsvm(test_repo);
  test.compute_dcgmax(10);

	k = ik;
  std::sort(k.begin(), k.end());
  k_max = k[k.size()-1];
}

void EvaluatorRating::evaluate(const Model& model) {
  double err = compute_pairwiseError(test, model);
  double ndcg = compute_ndcg(test, model);
  printf("%f, %f", err, ndcg);
}

struct pkcomp {
	bool operator() (std::pair<int, double> i, std::pair<int, double> j) {
		return i.second > j.second;
	}
};

struct vcomp {
	bool operator() (std::pair<int, double> i, std::pair<int, double> j) {
		return i.second < j.second;
	}
} vobj;

void EvaluatorBinary::load_files (const std::string& train_repo, const std::string& test_repo, std::vector<int>& ik) {
  std::cout << "load file" << std::endl;
	std::ifstream tr(train_repo);
	if (tr) {
		int uid, iid;
		while (tr >> uid >> iid) {
			if (train.size() < uid) train.resize(uid);
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
			if (test.size() < uid) test.resize(uid);
			test[uid - 1].insert(iid - 1);
		}
	} else {
		printf ("Error in opening the testing repository!\n");
		exit(EXIT_FAILURE);
	}
	te.close();

	k = ik;
  std::sort(k.begin(), k.end());
  k_max = k[k.size()-1];
} 

void EvaluatorBinary::evaluate (const Model& model) {
  vector<int> precision(k.size(), 0);

	#pragma omp parallel
	for (int i = 0; i < model.n_users; ++i) {
		std::priority_queue<std::pair<int, double>, std::vector<std::pair<int, double> >, pkcomp> pq;	
		for (int j = 0; j < model.n_items; ++j) {
			if (!train[i].empty() && train[i].find(j) != train[i].end()) {
				continue;
			}
			double score = 0;
			double *user_vec = &model.U[i * model.rank];
			double *item_vec = &model.V[j * model.rank];
			for (int l = 0; l < model.rank; ++l) {
				score += user_vec[l] * item_vec[l];
			}

			if (pq.size() < k_max) {
				pq.push(std::pair<int, double>(j, score));
			} else if (pq.top().second < score) {
				pq.push(std::pair<int, double>(j, score));
				pq.pop();	
			}
		}

    while(!pq.empty()) {
      int item = pq.top().first;
      if (!test[i].empty() && test[i].find(item) != test[i].end()) {
  			for(int j=k.size()-1; (j>=0) && (k[j]>=pq.size()); --j) {
          #pragma omp atomic
          ++precision[j];
        }
      }
			pq.pop();
		}
	}

  for(int l=0; l<k.size(); ++l) {
    printf("K%d: %f ", k[l], (double)precision[l] / (double)k[l] / model.n_users);
  }
}

void EvaluatorBinary::evaluateAUC(const Model& model) {
	double AUC = 0.;
	int num_users = model.n_users;
	#pragma omp parallel for reduction(+ : AUC)
	for (int i = 0; i < model.n_users; ++i) {
		std::vector<std::pair<int, double> > v;
		for (int j = 0; j < model.n_items; ++j) {
			if (!train[i].empty() && train[i].find(j) != train[i].end() ) {
				continue;
			}
			double score = 0;
			double *user_vec = &model.U[i * model.rank];
			double *item_vec = &model.V[j * model.rank];
			for (int l = 0; l < model.rank; ++l) {
				score += user_vec[l] * item_vec[l];
			}
			v.push_back(std::pair<int, double>(j, score) );		
		}
		std::sort(v.begin(), v.end(), vobj);

		int testNum = 0;
		int nonTestNum = 0;
		int accNumer = 0;
		for (int idx = 0; idx < v.size(); ++idx) {
			int j = v[idx].first;
			if (test[i].find(j) != test[i].end() ) {
				accNumer += nonTestNum;
				++testNum;
			} else {
				++nonTestNum;
			}
		}

		if (!testNum) {
			--num_users;
			continue;
		}

		AUC += accNumer * 1.0 / testNum / nonTestNum;
	}
	printf("AUC %f\n", AUC / num_users);
}
#endif
