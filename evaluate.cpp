#include <stdio.h>
#include <vector>
#include <utility>
#include <string>
#include "collrank.hpp"

int main (int argc, char* argv[]) {

  RatingMatrix TestRating, PredictedRating;

  std::string test_filename(argv[1]);
  std::string pred_filename(argv[2]);

  TestRating.read_lsvm(test_filename);
//  PredictedRating.read_lsvm(pred_filename);

  TestRating.compute_dcgmax(10);
  printf("NDCG : %f\n", compute_ndcg(TestRating, argv[2]));
/* 
  std::pair<double,double> pairwise_error = compute_pairwiseError(TestRating, PredictedRating);
  printf("KT distance : %f\n", pairwise_error.first);
  printf("Error for Top ratings : %f\n", pairwise_error.second);
*/
  return 0;

}
