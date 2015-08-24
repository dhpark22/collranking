#include <stdio.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>

#include "../code/elements.hpp"

enum extract_option_t { ALL, THRESHOLD, LARGEST_GAP };

class CompExtractor {
  protected:
    int n_users, n_items, n_ratings;

    std::vector<rating> ratings;
    std::ofstream train_file;
	  std::ofstream train_rating_lsvm, test_rating_lsvm;

    int n_train = 50, n_test = 10;

    inline int RandIDX(int, int);
    bool MakeComparisons(int, extract_option_t);
  
  public:
    CompExtractor(int nTrain, int nTest) : n_train(nTrain), n_test(nTest) {}
    bool Extract(char*, char*);
};

// Draw a random integer between (from) and (to-1) inclusively
inline int CompExtractor::RandIDX(int from, int to) { return (int)((double)rand() / (double)(RAND_MAX) * (double)(to-from)) + from; }

// Extract comparisons from ratings
bool CompExtractor::MakeComparisons(int new_user_id, extract_option_t extract_option) {

  int n_ratings_current_user = ratings.size();

  if (n_ratings_current_user >= n_train + n_test) {

    // Shuffle the ratings to split the ratings into training and test
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

    switch(extract_option) {
      case ALL :
        // Take all possible comparisons for the training ratings
        for(int j1=0; j1<n_train; j1++) {
          for(int j2=j1+1; j2<n_train; j2++) {
            if (ratings[j1].score > ratings[j2].score)
              train_file << new_user_id << ' ' << ratings[j1].item_id << ' ' << ratings[j2].item_id << std::endl;
            else if (ratings[j1].score < ratings[j2].score)
              train_file << new_user_id << ' ' << ratings[j2].item_id << ' ' << ratings[j1].item_id << std::endl;
          }
        }
        break;

      case THRESHOLD :
        // Take comparisons with gap > 1 for the training ratings
        for(int j1=0; j1<n_train; j1++) {
          for(int j2=j1+1; j2<n_train; j2++) {
            if (ratings[j1].score-1 > ratings[j2].score)
              train_file << new_user_id << ' ' << ratings[j1].item_id << ' ' << ratings[j2].item_id << std::endl;
            else if (ratings[j1].score+1 < ratings[j2].score)
              train_file << new_user_id << ' ' << ratings[j2].item_id << ' ' << ratings[j1].item_id << std::endl;
          }
        }
        break;

      case LARGEST_GAP :
        // Take the (n_train) comparisons with the largest rating differences 
        size_t n_comps = n_train;
        if (n_comps > comp_list.size()) n_comps = comp_list.size();
        for(int j=0; j<n_comps; j++) {
          train_file << new_user_id << ' ' << ratings[comp_list[j].user_id].item_id << 
                                       ' ' << ratings[comp_list[j].item_id].item_id << std::endl;
        }

    }

    // Write training ratings in lsvm format
    std::sort(ratings.begin(), ratings.begin()+n_train, rating_userwise);
    for(int j=0; j<n_train; ++j) {
      train_rating_lsvm << ratings[j].item_id << ':' << ratings[j].score << ' ';
    }
    train_rating_lsvm << std::endl;

    // Write test ratings in lsvm format
    std::sort(ratings.begin()+n_train, ratings.end(), rating_userwise);
    for(int j=n_train; j<ratings.size(); ++j) {
      test_rating_lsvm  << ratings[j].item_id << ':' << ratings[j].score << ' ';
    }
    test_rating_lsvm << std::endl;

    return true;
	}

	return false;
}

bool CompExtractor::Extract(char *input_filename, char *output_filename) {	
	
  // Open the files and read the metadata
	std::ifstream input_file;
	input_file.open(input_filename);
	if (!input_file.is_open()) { std::cerr << "File not opened!" << std::endl; return false; }

	input_file >> n_users >> n_items >> n_ratings;
  printf("%d users, %d items, %d ratings\n", n_users, n_items, n_ratings);	

  std::string output_header(output_filename);

  train_file.open(output_header + "_train_comps.dat");
  train_rating_lsvm.open(output_header + "_train_rating.lsvm");
  test_rating_lsvm.open (output_header + "_test_rating.lsvm");

  // Read the data
	std::string line;
	int user_id, item_id;
  double score;
	int current_user_id = 1, new_user_id = 1;

	for(int i=0; i<n_ratings; i++) {
		input_file >> user_id >> item_id >> score;

		if (current_user_id < user_id) {
			printf("User %d(%d)", current_user_id, ratings.size());
      if (MakeComparisons(new_user_id, ALL)) {
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
  if (MakeComparisons(new_user_id, ALL)) {
    std::cout << " - new id " << new_user_id << std::endl;
    ++new_user_id;
  }
  else std::cout << " dropped" << std::endl;

  --new_user_id;
	
	input_file.close();
	train_rating_lsvm.close();
  test_rating_lsvm.close();
  train_file.close();

	std::cout << "Comparisons for " << new_user_id << " users, " << n_items <<" items extracted" << std::endl;

  return true;
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


