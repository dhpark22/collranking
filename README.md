Preference Completion: Large-scale Collaborative Ranking from Pairwise Comparison

This repo contains the implementation of the following algorithms:
- Alternating SVM (AltSVM)
- Stochastic Gradient Descent (SGD)
- Global Ranking from All-aggregated pairwise comparisons 

### Compilation
On a UNIX-based system with a C++11 supporting compiler and OpenMP API, Compile using the Makefile
'''
$ make
'''

### Experiments on numerical ratings
Our trained model can be tested in terms of NDCG@10 when the test set consists of numerical ratings. To compare with other rating based algorithms, we provide a code that extracts pairwise comparisons from ratings. 

0. Prepare a dataset with (user, item, ratings) triple (example - data/movielens1m.txt)
1. Run util/num2comp.py to get training comparisons and test ratings. 
'''
$ python util/num2comp.py data/movielens1m.txt -o ml1m -n 50
'''
2. Set the configuration options with the files.
'''
train-file = ml1m-train.dat
train-file-rating = ml1m-train-ratings.lsvm 
test-file = ml1m-test-ratings.lsvm
'''
3. Run ./collrank
'''
$ ./collrank
'''

### Experiments on binary ratings
Our trained model can also be tested in terms of Precision@K when the test set consists of binary ratings.

