#include <stdio.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>

#include "elements.hpp"

class CompExtractor {
  protected:
    int n_users, n_items, n_ratings;

    std::vector<rating> ratings;
    std::ofstream train_file;
	  std::ofstream train_rating_lsvm, test_rating_lsvm;
    std::ofstream train_rating_prea, test_rating_prea; 

    std::vector<rating> train_ratings;
 
    int n_train = 50, n_test = 10;

    inline int RandIDX(int, int);
    bool MakeComparisons(int new_user_id);
  
  public:
    CompExtractor(int nTrain, int nTest) : n_train(nTrain), n_test(nTest) {}
    bool Extract(char*, char*);
};

// Draw a random integer between (from) and (to-1) inclusively
inline int CompExtractor::RandIDX(int from, int to) { return (int)((double)rand() / (double)(RAND_MAX) * (double)(to-from)) + from; }

// Extract comparisons from ratings
bool CompExtractor::MakeComparisons(int new_user_id) {

  int n_ratings_current_user = ratings.size();

  if (n_ratings_current_user >= n_train + n_test) {

    // Shuffle the ratings to split the ratings into training and test
	  //for(int j=0; j<n_train; j++) {
		//  int idx = RandIDX(j, n_ratings_current_user);
    //  ratings[j].swap(ratings[idx]);
    //}
    std::random_shuffle(ratings.begin(), ratings.end());

    std::vector<rating> comp_list(0);

	 	// Construct the whole comparison list for user i
	  for(int j1=0; j1<n_train; j1++) {
		  for(int j2=j1+1; j2<n_train; j2++) {
		  	if (ratings[j1].score > ratings[j2].score)
				  comp_list.push_back(rating(j1, j2, ratings[j1].score - ratings[j2].score));
			  else if (ratings[j1].score < ratings[j2].score)
				  comp_list.push_back(rating(j2, j1, ratings[j2].score - ratings[j1].score));
		  }
	  }

    // Sort the comparisons in the descending order of rating differences
    std::sort(comp_list.begin(), comp_list.end(), rating_scorewise);

    // Take the (n_train) comparisons with the largest rating differences 
    size_t n_comps = n_train;
    if (n_comps > comp_list.size()) n_comps = comp_list.size();
    for(int j=0; j<n_comps; j++) {
      train_file << new_user_id << ' ' << ratings[comp_list[j].user_id].item_id << 
                                   ' ' << ratings[comp_list[j].item_id].item_id << std::endl;
    }

/*
    // Take all possible comparisons for the training ratings
    for(int j1=0; j1<n_train; j1++) {
      for(int j2=j1+1; j2<n_train; j2++) {
					
        if (ratings[j1].score > ratings[j2].score)
          train_file << new_user_id << ' ' << ratings[j1].item_id << ' ' << ratings[j2].item_id << std::endl;
        else if (ratings[j1].score < ratings[j2].score)
          train_file << new_user_id << ' ' << ratings[j2].item_id << ' ' << ratings[j1].item_id << std::endl;

      }
    }
*/

    // Write training ratings in lsvm format
    std::sort(ratings.begin(), ratings.begin()+n_train, rating_userwise);
    for(int j=0; j<n_train; ++j) {
      train_rating_lsvm << ratings[j].item_id << ':' << ratings[j].score << ' ';
      train_ratings.push_back(rating(new_user_id, ratings[j].item_id, ratings[j].score));
    }
    train_rating_lsvm << std::endl;

    // Write test ratings in lsvm format
    std::sort(ratings.begin()+n_train, ratings.end(), rating_userwise);
    for(int j=n_train; j<ratings.size(); ++j) {
      test_rating_lsvm  << ratings[j].item_id << ':' << ratings[j].score << ' ';
      test_rating_prea  << new_user_id << '\t' << ratings[j].item_id << std::endl; 
      train_ratings.push_back(rating(new_user_id, ratings[j].item_id, ratings[j].score));
    }
    test_rating_lsvm << std::endl;

    return true;
	}

	return false;
}

bool CompExtractor::Extract(char *input_filename, char *output_filename) {	
	
  // Read the dataset
	std::ifstream input_file;
	input_file.open(input_filename);
	if (!input_file.is_open()) { std::cerr << "File not opened!" << std::endl; return false; }

	input_file >> n_users >> n_items >> n_ratings;
  printf("%d users, %d items, %d ratings\n", n_users, n_items, n_ratings);	

  std::string output_header(output_filename);

  train_file.open(output_header + "_train_comps.dat");

  train_rating_lsvm.open(output_header + "_train_rating.lsvm");
  train_rating_prea.open(output_header + "_train_rating_prea.arff");
  test_rating_lsvm.open (output_header + "_test_rating.lsvm");
  test_rating_prea.open (output_header + "_test_rating_prea.dat");

	std::string line;
	int user_id, item_id;
  double score;
	int current_user_id = 1, new_user_id = 1;

	for(int i=0; i<n_ratings; i++) {
		input_file >> user_id >> item_id >> score;

    if (i<5) printf("%d %d %f\n", user_id, item_id, score);
		//getline(input_file, line);

		if (current_user_id < user_id) {
			printf("User %d(%d)", current_user_id, ratings.size());
      if (MakeComparisons(new_user_id)) {
        printf(" - new id %d\n", new_user_id);
        new_user_id++;
      }
      else printf(" dropped\n");
			ratings.clear();
      current_user_id = user_id;
		}
		
		ratings.push_back(rating(user_id, item_id, score));
	}
			
  std::cout << "User " << user_id << "(" << ratings.size() << ")";
  if (MakeComparisons(new_user_id)) {
    std::cout << " - new id " << new_user_id << std::endl;
    ++new_user_id;
  }
  else std::cout << " dropped" << std::endl;

  --new_user_id;
	
  // write arff file for prea
  sort(train_ratings.begin(), train_ratings.end(), rating_userwise); 
  train_rating_prea << "@RELATION movievote" << std::endl << std::endl;
  train_rating_prea << "@ATTRIBUTE UserId NUMERIC" << std::endl;
  for(int iid=1; iid<=n_items; ++iid) {
    train_rating_prea << "@ATTRIBUTE 'Title " << iid << "' NUMERIC" << std::endl; 
  }
  for(int iid=1; iid<=n_items; ++iid) {
    train_rating_prea << "@ATTRIBUTE 'Date " << (n_items+iid) << "' STRING" << std::endl;
  }
  train_rating_prea << std::endl << "@DATA" << std::endl;
  int idx = 0;
  for(int uid=1; uid<=new_user_id; ++uid) {
    train_rating_prea << "{0 " << uid << ", ";
    while(train_ratings[idx].user_id == uid) {      
      train_rating_prea << train_ratings[idx].item_id << " " << train_ratings[idx].score << ", "; 
      ++idx;
    }
    train_rating_prea << (n_items+1) << " 0000-00-00}" << std::endl;
  }

	input_file.close();
	train_rating_lsvm.close();
  train_rating_prea.close();
  test_rating_lsvm.close();
  test_rating_prea.close();
  train_file.close();

	std::cout << "Comparisons for " << new_user_id << " users, " << n_items <<" items extracted" << std::endl;
}


int main(int argc, char **argv) {

	char *input_filename = nullptr, *output_filename = nullptr;

	if (argc < 2) {
		std::cout << "Extracting a random comparions dataset from a rating dataset" << std::endl;
		std::cout << "Usage   : ./extract_comps [options]" << std::endl;
		std::cout << "Options :  -n (number of training comparisons(or ratings) per user) " << std::endl;
		std::cout << "           -t (number of test comparisons per item) " << std::endl;
		std::cout << "           -i (input file name for rating dataset) " << std::endl; 
    std::cout << "           -o (header for output file names) " << std::endl; 

		return 0;
	}

  int n_train = 100, n_test = 10;

	for(int i=1; i<argc; i++) {
		if (argv[i][0] == '-') {
			switch(argv[i][1]) {
				case 'n':	// number for the sampling
					n_train = atoi(argv[++i]);
					std::cout << "# training ratings/user " << n_train << std::endl;
					break;

				case 't':
				  n_test = atoi(argv[++i]);
					std::cout << "# test ratings/user " << n_test << std::endl;
					break;

				case 'i':	// filename
					input_filename = argv[++i];
					std::cout << input_filename << std::endl;
					break;

				case 'o':
				  output_filename = argv[++i];
				  std::cout << output_filename << std::endl;
					break;
			}
		}
	}

  CompExtractor extractor(n_train, n_test); 
  if (!extractor.Extract(input_filename, output_filename)) { std::cerr << "Cannot extract!" << std::endl; exit(11); }

}


