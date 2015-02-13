#include <stdio.h>
#include <vector>
#include <utility>
#include <string>
#include "ratings.hpp"
#include "loss.hpp"

int main (int argc, char* argv[]) {

  if (argc < 2) {
    printf("Usage : ./evaluate [Test set in lsvm] [Predicted score in lsvm]\n");
  }

  RatingMatrix TestRating, PredictedRating;

  std::string test_filename(argv[1]);
  std::string pred_filename(argv[2]);

  TestRating.read_lsvm(test_filename);
  PredictedRating.read_lsvm(pred_filename);

  printf("KT distance : %f\n", compute_pairwiseError(TestRating, PredictedRating));

  TestRating.compute_dcgmax(10);
  printf("NDCG : %f\n", compute_ndcg(TestRating, PredictedRating));

  return 0;

}
