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
  public: 
  virtual void evaluate(const Model&) {}
  virtual void evaluateAUC(const Model&) {}

};



class EvaluatorBinary : public Evaluator {
  public:
    std::vector<std::unordered_set<int> > train, test;	
    std::vector<int> k;

    void load_files(char*, char*, std::vector<int>&);
    void evaluate(const Model&);
    void evaluateAUC(const Model&);
};

class EvaluatorRating : public Evaluator {

  RatingMatrix test;

  public:
    void load_files(char*);  
    void evaluate(const Model&);
};

void EvaluatorRating::load_files (char* test_ratings) {

  test.read_lsvm(test_ratings);
  test.compute_dcgmax(10);

}

void EvaluatorRating::evaluate(const Model& model) {
  
  double err = compute_pairwiseError(test, model);
  double ndcg = compute_ndcg(test, model);

  printf(" / %f %f ", err, ndcg);

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



void EvaluatorBinary::load_files (char* train_repo, char* test_repo, std::vector<int>& ik) {
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
} 

void EvaluatorBinary::evaluate (const Model& model) {
	int p1 = 0; 
	int p2 = 0;
	int p3 = 0;
	int p4 = 0;
	int p5 = 0;
	int p10 = 0;
	int p100 = 0;
	int p200 = 0;
	int p500 = 0;

	#pragma omp parallel for reduction(+ : p1, p2, p3, p4, p5, p10, p100, p200, p500)
	for (int i = 0; i < model.n_users; ++i) {
		std::priority_queue<std::pair<int, double>, std::vector<std::pair<int, double> >, pkcomp> pq;	
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

			if (pq.size() < 500) {
				pq.push(std::pair<int, double>(j, score) );
			} else if (pq.top().second < score) {
				pq.push(std::pair<int, double>(j, score) );
				pq.pop();	
			}
		}

		int ps = pq.size();
		while (ps) {
			int item = pq.top().first;
			if (!test[i].empty() && test[i].find(item) != test[i].end() ) {
				if (ps < 501) ++p500;
				if (ps < 201) ++p200;
				if (ps < 101) ++p100;
				if (ps < 11) ++p10;
				if (ps < 6) ++p5;
				if (ps < 5) ++p4;
				if (ps < 4) ++p3;
				if (ps < 3) ++p2;
				if (ps < 2) ++p1;
			}			
			pq.pop();
			--ps;
		}
	}

	printf("compute precision at k\n");
	printf("k %d, precision %f\n", 1, (double) p1 / model.n_users);
	printf("k %d, precision %f\n", 2, (double) p2 / 2 / model.n_users);
	printf("k %d, precision %f\n", 3, (double) p3 / 3 / model.n_users);
	printf("k %d, precision %f\n", 4, (double) p4 / 4 / model.n_users);
	printf("k %d, precision %f\n", 5, (double) p5 / 5 / model.n_users);
	printf("k %d, precision %f\n", 10, (double) p10 / 10 / model.n_users);
	printf("k %d, precision %f\n", 100, (double) p100 / 100 / model.n_users);
	printf("k %d, precision %f\n", 200, (double) p200 / 200 / model.n_users);
	printf("k %d, precision %f\n", 500, (double) p500 / 500 / model.n_users);
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
