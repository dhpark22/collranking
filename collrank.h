#ifndef _COLLRANK_H
#define _COLLRANK_H

#include <utility>
#include <string>
#include <random>
#include <functional>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <algorithm>
#include <map>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>

using namespace std;

struct rating
{
	int user_id;
	int item_id;
	int score;

	rating(): user_id(0), item_id(0), score(0) {}
	rating(int u, int i, int s): user_id(u), item_id(i), score(s) {}
	void setvalues(const int u, const int i, const int s) {
		user_id = u;
		item_id = i;
		score = s;
	}
	void swap(rating& r) {
		int temp;
		temp = user_id; user_id = r.user_id; r.user_id = temp;
		temp = item_id; item_id = r.item_id; r.item_id = temp;
		temp = score; score = r.score; r.score = temp;
	}
};

struct ratingf
{
	int user_id;
	int item_id;
	double score;

	ratingf(): user_id(0), item_id(0), score(0.) {}
	ratingf(int u, int i, double s): user_id(u), item_id(i), score(s) {}
	void setvalues(const int u, const int i, const double s) {
		user_id = u;
		item_id = i;
		score = s;
	}
	void swap(ratingf& r) {
		int temp;
		temp = user_id; user_id = r.user_id; r.user_id = temp;
		temp = item_id; item_id = r.item_id; r.item_id = temp;
    double tempf;
		tempf = score; score = r.score; r.score = tempf;
	}
};

struct comparison
{
	int user_id;
	int item1_id;
	int item2_id;
  int comp;

	comparison(): user_id(0), item1_id(0), item2_id(0), comp(1) {}
	comparison(int u, int i1, int i2, int cp): user_id(u), item1_id(i1), item2_id(i2), comp(cp) {}
	comparison(const comparison& c): user_id(c.user_id), item1_id(c.item1_id), item2_id(c.item2_id), comp(c.comp) {}
	void setvalues(const int u, const int i1, const int i2, const int cp) {
		user_id = u;
		item1_id = i1;
		item2_id = i2;
    comp = cp;
	}
	void swap(comparison& c) {
		int temp;
		temp = user_id; user_id = c.user_id; c.user_id = temp;
		temp = item1_id; item1_id = c.item1_id; c.item1_id = temp;
		temp = item2_id; item2_id = c.item2_id; c.item2_id = temp;
	  temp = comp; comp = c.comp; c.comp = temp;
  }
};

bool comp_userwise(comparison a, comparison b) { return ((a.user_id < b.user_id) || ((a.user_id == b.user_id) && (a.item1_id < b.item1_id))); }
bool comp_itemwise(comparison a, comparison b) { return ((a.item1_id < b.item1_id) || ((a.item1_id == b.item1_id) && (a.user_id < b.user_id))); }

bool comp_ratingwise(rating a, rating b) { return (a.score > b.score); }
bool rate_userwise(rating a, rating b) { return ((a.user_id < b.user_id) || ((a.user_id == b.user_id) && (a.item_id < b.item_id))); }
bool ratef_userwise(ratingf a, ratingf b) { return ((a.user_id < b.user_id) || ((a.user_id == b.user_id) && (a.item_id < b.item_id))); }
bool ratef_ratingwise(ratingf a, ratingf b) { return (a.score > b.score); }

typedef struct rating rating;
typedef struct ratingf ratingf;
typedef struct comparison comparison;

template <typename T>
class RatingMatrix {
  public:
    int             n_users, n_items;
    vector<ratingf> ratings;
    vector<int>     idx;

    int             ndcg_k = 0;
    bool            is_dcg_max_computed = false;
    vector<double>  dcg_max;
    void compute_dcgmax(int);

    RatingMatrix() : n_users(0), n_items(0) {}
    RatingMatrix(int nu, int ni): n_users(nu), n_items(ni) {}

    void read_lsvm(const string&);
    void read_spformat(const string&);
};

class Model {
  public:
    bool is_allocated;
    int n_users, n_items; 	// number of users/items in training sample, number of samples in traing and testing data set
    int rank;                       // parameters
    double *U, *V;                                  // low rank U, V

    void allocate(int nu, int ni);    
    void de_allocate();					            // deallocate U, V when they are used multiple times by different methods

    Model(int r): is_allocated(false), rank(r) {}
 
    double Unormsq();
    double Vnormsq();
    double compute_loss(const vector<comparison>&, double);
    double compute_testerror(const vector<comparison>&);
};

std::pair<double,double> compute_pairwiseError(const RatingMatrix<double>& TestRating, const RatingMatrix<double>& PredictedRating) {

  std::pair<double,double> comp_error;
  std::vector<double> score(TestRating.n_items);
  
  int error = 0, n_comps = 0, errorT = 0, n_compsT = 0;
  for(int uid=0; uid<TestRating.n_users; ++uid) {
    score.resize(TestRating.n_items,-1e10);    
    double max_sc = -1.;

    int j = PredictedRating.idx[uid];
    for(int i=TestRating.idx[uid]; i<TestRating.idx[uid+1]; ++i) {
      int iid = TestRating.ratings[i].item_id;
      while((j < PredictedRating.idx[uid+1]) && (PredictedRating.ratings[j].item_id < iid)) ++j;
      if ((PredictedRating.ratings[j].user_id == uid) && (PredictedRating.ratings[j].item_id == iid)) score[iid] = PredictedRating.ratings[j].score;
      if (TestRating.ratings[i].score > max_sc) max_sc = TestRating.ratings[i].score;
    }

    max_sc = max_sc - .1;

    for(int i=TestRating.idx[uid]; i<TestRating.idx[uid+1]-1; ++i) {
      for(int j=i+1; j<TestRating.idx[uid+1]; ++j) {
        int item1_id = TestRating.ratings[i].item_id;
        int item2_id = TestRating.ratings[j].item_id; 
        double item1_sc = TestRating.ratings[i].score;
        double item2_sc = TestRating.ratings[j].score;      

        if ((item1_sc > item2_sc) && (score[item1_id] <= score[item2_id])) ++error;
        if ((item1_sc < item2_sc) && (score[item1_id] >= score[item2_id])) ++error;
        ++n_comps;

        if ((item1_sc >= max_sc) && (item2_sc < max_sc)) {
          if (score[item1_id] <= score[item2_id]) ++errorT;
          ++n_compsT;
        }

        if ((item2_sc >= max_sc) && (item1_sc < max_sc)) {
          if (score[item2_id] >= score[item1_id]) ++errorT;
          ++n_compsT;
        }
    
      }
    }
  }

  comp_error.first  = (double)error / (double)n_comps;
  comp_error.second = (double)errorT / (double)n_compsT; 

  printf("%d %d %d %d \n", error, n_comps, errorT, n_compsT);

  return comp_error;
}

std::pair<double,double> compute_pairwiseError(const RatingMatrix<double>& TestRating, const Model& PredictedModel) {

  std::pair<double,double> comp_error;
  std::vector<double> score(TestRating.n_items);
  
  int error = 0, n_comps = 0, errorT = 0, n_compsT = 0;
  for(int uid=0; uid<TestRating.n_users; ++uid) {
    score.resize(TestRating.n_items,-1e10);    
    double max_sc = -1.;

    for(int i=TestRating.idx[uid]; i<TestRating.idx[uid+1]; ++i) {
      int iid = TestRating.ratings[i].item_id;
      double prod = 0.;
      for(int k=0; k<PredictedModel.rank; ++k) prod += PredictedModel.U[uid * PredictedModel.rank + k] * PredictedModel.V[iid * PredictedModel.rank + k];
      score[iid] = prod;
      if (TestRating.ratings[i].score > max_sc) max_sc = TestRating.ratings[i].score;
    }

    max_sc = max_sc - .1;

    for(int i=TestRating.idx[uid]; i<TestRating.idx[uid+1]-1; ++i) {
      for(int j=i+1; j<TestRating.idx[uid+1]; ++j) {
        int item1_id = TestRating.ratings[i].item_id;
        int item2_id = TestRating.ratings[j].item_id; 
        double item1_sc = TestRating.ratings[i].score;
        double item2_sc = TestRating.ratings[j].score;      

        if ((item1_sc > item2_sc) && (score[item1_id] <= score[item2_id])) ++error;
        if ((item1_sc < item2_sc) && (score[item1_id] >= score[item2_id])) ++error;
        ++n_comps;

        if ((item1_sc >= max_sc) && (item2_sc < max_sc)) {
          if (score[item1_id] <= score[item2_id]) ++errorT;
          ++n_compsT;
        }

        if ((item2_sc >= max_sc) && (item1_sc < max_sc)) {
          if (score[item2_id] >= score[item1_id]) ++errorT;
          ++n_compsT;
        }
    
      }
    }
  }

  comp_error.first  = (double)error / (double)n_comps;
  comp_error.second = (double)errorT / (double)n_compsT; 

  return comp_error;
}

double compute_ndcg(const RatingMatrix<double>& TestRating, const RatingMatrix<double>& PredictedRating) {

  double ndcg_sum = 0.;
  vector<pair<double,int> > ranking;
 
  for(int uid=0; uid<TestRating.n_users; ++uid) {
    double dcg = 0.;
 
    ranking.clear();
    int j = PredictedRating.idx[uid];
    for(int i=TestRating.idx[uid]; i<TestRating.idx[uid+1]; ++i) {
      double prod = 0.;
      while((j < PredictedRating.idx[uid+1]) && (PredictedRating.ratings[j].item_id < TestRating.ratings[i].item_id)) ++j;
      if ((PredictedRating.ratings[j].user_id == TestRating.ratings[i].user_id) && (PredictedRating.ratings[j].item_id == TestRating.ratings[i].item_id))
        ranking.push_back(pair<double,int>(PredictedRating.ratings[j].score,0));
      else
        ranking.push_back(pair<double,int>(-1e10,0));
    }

    double min_score = ranking[0].first;
    for(int j=0; j<ranking.size(); ++j) min_score = min(min_score, ranking[j].first);

    for(int k=1; k<=TestRating.ndcg_k; ++k) {
      int topk_idx = -1;
      double max_score = min_score - 1.;
      for(int j=0; j<ranking.size(); ++j) {
        if ((ranking[j].second == 0) && (ranking[j].first > max_score)) {
          max_score = ranking[j].first;
          topk_idx = j;
        }
      }
      ranking[topk_idx].second = k;
     
      dcg += (double)(pow(2,TestRating.ratings[TestRating.idx[uid]+topk_idx].score) - 1) / log2((double)(k+1));
    }
   
    ndcg_sum += dcg / TestRating.dcg_max[uid];
  }

  return ndcg_sum / (double)PredictedRating.n_users;
}

double compute_ndcg(const RatingMatrix<double>& TestRating, const Model& PredictedModel) {
  
  double ndcg_sum = 0.;
  vector<pair<double,int> > ranking;
 
  for(int uid=0; uid<PredictedModel.n_users; ++uid) {
    double dcg = 0.;
  
    ranking.clear();
    for(int i=TestRating.idx[uid]; i<TestRating.idx[uid+1]; ++i) {
      int iid = TestRating.ratings[i].item_id;
      double prod = 0.;
      for(int k=0; k<PredictedModel.rank; ++k) prod += PredictedModel.U[uid * PredictedModel.rank + k] * PredictedModel.V[iid * PredictedModel.rank + k];
      ranking.push_back(pair<double,int>(prod,0));
    }

    double min_score = ranking[0].first;
    for(int j=0; j<ranking.size(); ++j) min_score = min(min_score, ranking[j].first);

    for(int k=1; k<=TestRating.ndcg_k; ++k) {
      int topk_idx = -1;
      double max_score = min_score - 1.;
      for(int j=0; j<ranking.size(); ++j) {
        if ((ranking[j].second == 0) && (ranking[j].first > max_score)) {
          max_score = ranking[j].first;
          topk_idx = j;
        }
      }
      ranking[topk_idx].second = k;
     
      dcg += (double)(pow(2,TestRating.ratings[TestRating.idx[uid]+topk_idx].score) - 1) / log2((double)(k+1));
    }
   
   ndcg_sum += dcg / TestRating.dcg_max[uid];
	}

  return ndcg_sum / (double)PredictedModel.n_users;
}

template <typename T>
void RatingMatrix<T>::read_lsvm(const string& filename) {

  ratings.clear();
  idx.clear();

  n_users = 0;
  n_items = 0;

  // Read ratings file for NDCG
  vector<ratingf> ratings_current_user(0);
  string user_str, attribute_str;
  stringstream attribute_sstr;  

  ifstream f;
 
  f.open(filename);
  if (f.is_open()) {
    int    uid = 0, iid;
    double sc;
    idx.push_back(0);
      
    while(1) {
      getline(f, user_str);

      size_t pos1 = 0, pos2;
      while(1) {
        pos2 = user_str.find(':', pos1); if (pos2 == string::npos) break; 
        attribute_str = user_str.substr(pos1, pos2-pos1);
        attribute_sstr.clear(); attribute_sstr.str(attribute_str);
        attribute_sstr >> iid;
        n_items = max(n_items, iid);
        --iid;
        pos1 = pos2+1;

        pos2 = user_str.find(' ', pos1); attribute_str = user_str.substr(pos1, pos2-pos1);
        attribute_sstr.clear(); attribute_sstr.str(attribute_str);
        attribute_sstr >> sc;
        pos1 = pos2+1;

        ratings_current_user.push_back(ratingf(uid, iid, sc));
      }
      if (ratings_current_user.size() == 0) break;
         
      sort(ratings_current_user.begin(), ratings_current_user.end(), ratef_userwise);
 
      for(int j=0; j<ratings_current_user.size(); ++j) ratings.push_back(ratings_current_user[j]);
      idx.push_back(ratings.size());
    
      ratings_current_user.clear();
 
      ++uid;
    }

    n_users = uid;

    printf("Read dataset with %d users, %d items \n", n_users, n_items);

  } else {
      printf("Error in opening the extracted rating file!\n");
      exit(EXIT_FAILURE);
  }
  
  f.close();
}

template <typename T>
void RatingMatrix<T>::compute_dcgmax(int ndcgK) {

  ndcg_k = ndcgK;

  vector<ratingf> ratings_current_user(0);

  dcg_max.resize(n_users, 0.);  
  for(int uid=0; uid<n_users; ++uid) {
    for(int i=idx[uid]; i<idx[uid+1]; ++i) ratings_current_user.push_back(ratings[i]);
    
    sort(ratings_current_user.begin(), ratings_current_user.end(), ratef_ratingwise);
    
    for(int k=0; k<ndcg_k; ++k) dcg_max[uid] += (double)(pow(2,ratings_current_user[k].score) - 1.) / log2(k+2); 
    
    ratings_current_user.clear();
  } 
}

template <typename T>
void RatingMatrix<T>::read_spformat(const string& filename) {
  // read sparse matrix format
}

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

double Model::compute_loss(const vector<comparison>& TestComps, double lambda) {
  double p = 0., slack;
  for(int i=0; i<TestComps.size(); ++i) {
    double *user_vec  = &U[TestComps[i].user_id  * rank];
    double *item1_vec = &V[TestComps[i].item1_id * rank];
    double *item2_vec = &V[TestComps[i].item2_id * rank];
    double d = 0.;
    for(int j=0; j<rank; ++j) {
      d += user_vec[j] * (item1_vec[j] - item2_vec[j]);
    }
    slack = max(0., 1. - d);
    p += slack*slack/lambda;
  }
   
  return p;		
}

double Model::compute_testerror(const vector<comparison>& TestComps) {
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
