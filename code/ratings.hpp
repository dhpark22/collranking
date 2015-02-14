#ifndef __RATINGS_H__
#define __RATINGS_H__

#include <string>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <algorithm>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>

#include "elements.hpp"
#include "model.hpp"

class RatingMatrix {
  public:
    int                   n_users, n_items;
    std::vector<rating>   ratings;
    std::vector<int>      idx;

    int                   ndcg_k = 0;
    bool                  is_dcg_max_computed = false;
    std::vector<double>   dcg_max;
    
    void compute_dcgmax(int);
    double compute_user_ndcg(int, const std::vector<double>&) const;

    RatingMatrix() : n_users(0), n_items(0) {}
    RatingMatrix(int nu, int ni): n_users(nu), n_items(ni) {}

    void read_lsvm(const std::string&);
    void read_spformat(const std::string&);
};

std::ifstream::pos_type filesize(const char* filename)
{
    std::ifstream in(filename, std::ios::binary | std::ios::ate);
    return in.tellg(); 
}

double RatingMatrix::compute_user_ndcg(int uid, const std::vector<double>& score) const {
  std::vector<std::pair<double,int> > ranking;
  for(int j=0; j<score.size(); ++j) ranking.push_back(std::pair<double,int>(score[j],0));

  double min_score = ranking[0].first;
  for(int j=0; j<ranking.size(); ++j) min_score = std::min(min_score, ranking[j].first);

  double dcg = 0.;
  for(int k=1; k<=ndcg_k; ++k) {
    int topk_idx = -1;
    double max_score = min_score - 1.;
    for(int j=0; j<ranking.size(); ++j) {
      if ((ranking[j].second == 0) && (ranking[j].first > max_score)) {
        max_score = ranking[j].first;
        topk_idx = j;
      }
    }
    ranking[topk_idx].second = k;
     
    dcg += (double)(pow(2,ratings[idx[uid]+topk_idx].score) - 1) / log2((double)(k+1));
  }

  return dcg / dcg_max[uid];
} 

void RatingMatrix::read_lsvm(const std::string& filename) {
  
  ratings.clear();
  idx.clear();

  n_users = 0;
  n_items = 0;

  // Read ratings file for NDCG
  std::vector<rating> ratings_current_user(0);
  std::string user_str, attribute_str;
  std::stringstream attribute_sstr;  

  std::ifstream f;
 
  f.open(filename);
  if (f.is_open()) {
    int    uid = 0, iid;
    double sc;
    idx.push_back(0);
      
    while(1) {
      getline(f, user_str);

      size_t pos1 = 0, pos2;
      while(1) {
        pos2 = user_str.find(':', pos1); if (pos2 == std::string::npos) break; 
        attribute_str = user_str.substr(pos1, pos2-pos1);
        attribute_sstr.clear(); attribute_sstr.str(attribute_str);
        attribute_sstr >> iid;
        n_items = std::max(n_items, iid);
        --iid;
        pos1 = pos2+1;

        pos2 = user_str.find(' ', pos1); attribute_str = user_str.substr(pos1, pos2-pos1);
        attribute_sstr.clear(); attribute_sstr.str(attribute_str);
        attribute_sstr >> sc;
        pos1 = pos2+1;

        ratings_current_user.push_back(rating(uid, iid, sc));
      }
      if (ratings_current_user.size() == 0) break;
         
      sort(ratings_current_user.begin(), ratings_current_user.end(), rating_userwise);
 
      for(int j=0; j<ratings_current_user.size(); ++j) ratings.push_back(ratings_current_user[j]);
      idx.push_back(ratings.size());
    
      ratings_current_user.clear();
 
      ++uid;
    }

    n_users = uid;

    printf("%d users, %d items \n", n_users, n_items);

  } else {
      printf("Error in opening the extracted rating file!\n");
      std::cout << filename << std::endl;
      exit(EXIT_FAILURE);
  }
  
  f.close();
}

void RatingMatrix::compute_dcgmax(int ndcgK) {

  ndcg_k = ndcgK;

  std::vector<rating> ratings_current_user(0);

  dcg_max.resize(n_users, 0.);  
  for(int uid=0; uid<n_users; ++uid) {
    for(int i=idx[uid]; i<idx[uid+1]; ++i) ratings_current_user.push_back(ratings[i]);
    
    std::sort(ratings_current_user.begin(), ratings_current_user.end(), rating_scorewise);
    
    for(int k=1; k<=ndcg_k; ++k) dcg_max[uid] += (double)(pow(2,ratings_current_user[k-1].score) - 1.) / log2(k+1); 
    
    ratings_current_user.clear();
  }

  is_dcg_max_computed = true; 
}

void RatingMatrix::read_spformat(const std::string& filename) {
  // read sparse matrix format
}

#endif
