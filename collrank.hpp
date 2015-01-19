#ifndef __COLLRANK_H__
#define __COLLRANK_H__

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

#include "elements.hpp"
#include "model.hpp"
#include "ratings.hpp"

using namespace std;

std::pair<double,double> compute_pairwiseError(const RatingMatrix& TestRating, const RatingMatrix& PredictedRating) {

  std::pair<double,double> comp_error;
  std::vector<double> score(TestRating.n_items);
  
  unsigned long long error = 0, n_comps = 0, errorT = 0, n_compsT = 0;
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

  return comp_error;
}

std::pair<double,double> compute_pairwiseError(const RatingMatrix& TestRating, const Model& PredictedModel) {

  std::pair<double,double> comp_error;
  std::vector<double> score(TestRating.n_items);
  
  unsigned long long error = 0, n_comps = 0, errorT = 0, n_compsT = 0;
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

double compute_ndcg(RatingMatrix& TestRating, const string& Predict_filename) {

  double ndcg_sum;
 
  vector<double> score;
  string user_str, attribute_str;
  stringstream attribute_sstr;  

  ifstream f;
 
  f.open(Predict_filename);
  if (f.is_open()) {
     
    for(int uid=0; uid<TestRating.n_users; ++uid) { 
      getline(f, user_str);

      size_t pos1 = 0, pos2;
     
      score.clear();
      for(int idx=TestRating.idx[uid]; idx<TestRating.idx[uid+1]; ++idx) {
        int iid = -1;
        double sc;

        while(iid < TestRating.ratings[idx].item_id) {
          pos2 = user_str.find(':', pos1); if (pos2 == string::npos) break; 
          attribute_str = user_str.substr(pos1, pos2-pos1);
          attribute_sstr.clear(); attribute_sstr.str(attribute_str);
          attribute_sstr >> iid;
          --iid;
          pos1 = pos2+1;

          pos2 = user_str.find(' ', pos1); attribute_str = user_str.substr(pos1, pos2-pos1);
          attribute_sstr.clear(); attribute_sstr.str(attribute_str);
          attribute_sstr >> sc;
          pos1 = pos2+1;
        }
      
        if (iid == TestRating.ratings[idx].item_id)
          score.push_back(sc);
        else
          score.push_back(-1e10);

      }         
 
      ndcg_sum += TestRating.compute_user_ndcg(uid, score);
    } 

  } else {
      printf("Error in opening the extracted rating file!\n");
      cout << Predict_filename << endl;
      exit(EXIT_FAILURE);
  }
  
  f.close();
}

double compute_ndcg(RatingMatrix& TestRating, const RatingMatrix& PredictedRating) {

  double ndcg_sum = 0.;
  vector<double> score; 
 
  for(int uid=0; uid<TestRating.n_users; ++uid) {
    double dcg = 0.;
 
    score.clear();
    int j = PredictedRating.idx[uid];
    for(int i=TestRating.idx[uid]; i<TestRating.idx[uid+1]; ++i) {
      double prod = 0.;
      while((j < PredictedRating.idx[uid+1]) && (PredictedRating.ratings[j].item_id < TestRating.ratings[i].item_id)) ++j;
      if ((PredictedRating.ratings[j].user_id == TestRating.ratings[i].user_id) && (PredictedRating.ratings[j].item_id == TestRating.ratings[i].item_id))
        score.push_back(PredictedRating.ratings[j].score);
      else
        score.push_back(-1e10);
    }
  
    ndcg_sum += TestRating.compute_user_ndcg(uid, score);
  }

  return ndcg_sum / (double)PredictedRating.n_users;
}

double compute_ndcg(RatingMatrix& TestRating, const Model& PredictedModel) {
  
  double ndcg_sum = 0.;
  vector<double> score;
 
  for(int uid=0; uid<PredictedModel.n_users; ++uid) {
    double dcg = 0.;
  
    score.clear();
    for(int i=TestRating.idx[uid]; i<TestRating.idx[uid+1]; ++i) {
      int iid = TestRating.ratings[i].item_id;
      double prod = 0.;
      for(int k=0; k<PredictedModel.rank; ++k) prod += PredictedModel.U[uid * PredictedModel.rank + k] * PredictedModel.V[iid * PredictedModel.rank + k];
      score.push_back(prod);
    }
    
    ndcg_sum += TestRating.compute_user_ndcg(uid, score);
	}

  return ndcg_sum / (double)PredictedModel.n_users;
}

#endif
