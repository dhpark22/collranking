#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include "collrank.h"

using namespace std;

int n_users, n_items, n_ratings;

bool SAMPLE_BY_COMPARISONS = true;
int n_train = 100, n_test = 100;

// take a random integer between (from) and (to-1)
inline int randidx(int from, int to) { return (int)((double)rand() / (double)(RAND_MAX) * (double)(to-from)) + from; }

bool makeComparisons(vector<rating> &ratings, 
					 ofstream &output_file, ofstream &train_file, ofstream &test_file,
					 int new_user_id, int n_train, int n_test) {

	int n_ratings_current_user = ratings.size();

	if (SAMPLE_BY_COMPARISONS) {
		vector<comparison> comp_list(0);
		// comparison comp;

		// Construct the whole comparison list for user i
		for(int j1=0; j1<n_ratings_current_user; j1++) {
			for(int j2=j1+1; j2<n_ratings_current_user; j2++) {
						
				if (ratings[j1].score > ratings[j2].score)
					// comp.setvalues(0, ratings[j1].item_id, ratings[j2].item_id);
					comp_list.push_back(comparison(0, ratings[j1].item_id, ratings[j2].item_id, 1));
				else if (ratings[j2].score < ratings[j1].score)
					// comp.setvalues(0, ratings[j2].item_id, ratings[j1].item_id);
					comp_list.push_back(comparison(0, ratings[j2].item_id, ratings[j1].item_id, 1));

			}
		}

		// Subsample comparisons only for the users with enough comparisons
		if (comp_list.size() >= n_train + n_test) {

      // Write test ratings for this user to output_file
      for(int j=0; j<n_ratings_current_user; ++j) {
        output_file << new_user_id << ' ' << ratings[j].item_id << ' ' << ratings[j].score << endl;
      }

			// Subsample comparisons for training 
			for(int j=0; j<n_train; j++) {
				int idx = randidx(j, comp_list.size());

				if (comp_list[idx].item1_id == 0) {
					cout << new_user_id << ' ' << j << ' ' << comp_list.size() << ' ' << idx << endl;
				}

				train_file << new_user_id << ' ' << comp_list[idx].item1_id << ' ' << comp_list[idx].item2_id << endl;
				comp_list[j].swap(comp_list[idx]);
			}

			// Subsample comparisons for test
			for(int j=n_train; j<n_train+n_test; j++) {
				int idx = randidx(j, comp_list.size());
				test_file << new_user_id << ' ' << comp_list[idx].item1_id << ' ' << comp_list[idx].item2_id << endl;
				comp_list[j].swap(comp_list[idx]);
			}

			return true;
		}
		else
			return false;
	}
	else {

		if (n_ratings_current_user >= n_train + n_test) {

			for(int j=0; j<n_train+n_test; j++) {
				int idx = randidx(j, n_ratings_current_user);
				ratings[j].swap(ratings[idx]);
			}

			for(int j1=0; j1<n_train; j1++) {
				for(int j2=j1+1; j2<n_train; j2++) {
					
					if (ratings[j1].score > ratings[j2].score)
						train_file << new_user_id << ' ' << ratings[j1].item_id << ' ' << ratings[j2].item_id << endl;
					else if (ratings[j1].score < ratings[j2].score)
						train_file << new_user_id << ' ' << ratings[j2].item_id << ' ' << ratings[j1].item_id << endl;

				}
			}

      // Write test ratings for this user to output_file
      for(int j=n_train; j<n_train+n_test; ++j) {
        output_file << new_user_id << ' ' << ratings[j].item_id << ' ' << ratings[j].score << endl;
      }

			for(int j1=n_train; j1<n_train+n_test; j1++) {
				for(int j2=j1+1; j2<n_train+n_test; j2++) {
							
					if (ratings[j1].score > ratings[j2].score)
						test_file << new_user_id << ' ' << ratings[j1].item_id << ' ' << ratings[j2].item_id << endl;
					else if (ratings[j1].score < ratings[j2].score)
						test_file << new_user_id << ' ' << ratings[j2].item_id << ' ' << ratings[j1].item_id << endl;

				}
			}

			return true;
		}
		else
			return false;
	}

	return false;
}

int main(int argc, char **argv) {

	char *input_filename = nullptr, *output_filename = nullptr;
  char *train_filename = nullptr, *test_filename = nullptr;

	if (argc < 2) {
		cout << "Extracting a random comparions dataset from a rating dataset" << endl;
		cout << "Usage   : ./extract_comps [options]" << endl;
		cout << "Options :  -n (number of training comparisons per user) " << endl;
		cout << "           -t (number of test comparisons per item) " << endl;
		cout << "           -i (input file name for rating dataset) " << endl; 
		cout << "           -o0 (output file name for extracted users) " << endl;
    cout << "           -o1 (output file name for traning comparisons) " << endl; 
		cout << "           -o2 (output file name for test comparisons) " << endl; 

		return 0;
	}

	for(int i=1; i<argc; i++) {
		if (argv[i][0] == '-') {
			switch(argv[i][1]) {
				case 'c':	// sampling items
					SAMPLE_BY_COMPARISONS = false;
					break;

				case 'n':	// number for the sampling
					n_train = atoi(argv[++i]);
					cout << "Training sample size " << n_train << endl;
					break;

				case 't':
					n_test = atoi(argv[++i]);
					cout << "Test sample size " << n_test << endl;
					break;

				case 'i':	// filename
					input_filename = argv[++i];
					cout << input_filename << endl;
					break;

				case 'o':
					switch(argv[i][2]) {
            case '0':
            output_filename = argv[++i];
            cout << output_filename << endl;
            break;

						case '1':
						train_filename = argv[++i];
						cout << train_filename << endl;
						break;

						case '2':
						test_filename = argv[++i];
						cout << test_filename << endl;
						break;
					}

					break;
			}
		}
	}

	if (input_filename == nullptr) { cerr << "Input file required!" << endl; exit(11); }
	
	// Read the dataset
	ifstream input_file;
	input_file.open(input_filename);
	if (!input_file.is_open()) { cerr << "File not opened!" << endl; exit(11); }

	input_file >> n_users >> n_items >> n_ratings;
	cout << n_users << " users, " << n_items << " items, " << n_ratings << " ratings" << endl;

	ofstream output_file, train_file, test_file;
	if (output_filename == nullptr) output_file.open("rating_extracted.rat"); else output_file.open(output_filename);
	if (train_filename == nullptr) train_file.open("comp_train.cmp"); else train_file.open(train_filename);
	if (test_filename == nullptr) test_file.open("comp_test.cmp"); else test_file.open(test_filename);

	string line;
	rating r;
	vector<rating> ratings(0);
	srand(time(NULL));

	int current_user_id = 1, new_user_id = 1;

	for(int i=0; i<n_ratings; i++) {
		input_file >> r.user_id >> r.item_id >> r.score;
		getline(input_file, line);

		if (current_user_id < r.user_id) {
			if (makeComparisons(ratings, output_file, train_file, test_file, new_user_id, n_train, n_test)) new_user_id++;
			ratings.clear();
			current_user_id = r.user_id;
		}
		
		ratings.push_back(r);
	}
	if (makeComparisons(ratings, output_file, train_file, test_file, new_user_id, n_train, n_test)) new_user_id++;

	new_user_id--;

	input_file.close();
	train_file.close();
	test_file.close();

	cout << "Comparisons for " << new_user_id << " users extracted" << endl;
}
