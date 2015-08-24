### Preference Completion: Large-scale Collaborative Ranking from Pairwise Comparison

This repo contains the implementation of the following algorithms:
- Alternating SVM (AltSVM)
- Stochastic Gradient Descent (SGD)
- Global Ranking from All-aggregated pairwise comparisons 

#### Compilation
On a UNIX-based system with a C++11 supporting compiler and OpenMP API, compile using the Makefile
```
$ make
```

#### Experiments on numerical ratings
Our trained model can be tested in terms of NDCG@10 when the test set consists of numerical ratings.

For comparison with other rating based methods, we provide a Python script (util/num2comp.py) that divides a (user, item, rating) dataset into a training set and a test set, and extract pairwise comparisons from the training set. 

0. Prepare a dataset with (user, item, ratings) triple. (Example: data/movielens1m.txt)

1. Run util/num2comp.py to get training comparisons and test ratings. 
```
$ python util/num2comp.py data/movielens1m.txt -o ml1m -n 50
```
(The script also generates the training ratings which can be used for other methods)

2. Set the configuration options.
```
[input]
type = numeric
train\_file = ml1m\_train.dat
test\_file = ml1m\_test\_ratings.lsvm
```

3. Run the binary. 
```
$ ./collrank
```

#### Experiments on binary ratings
Our trained model can also be tested in terms of Precision@K when the test set consists of binary ratings.

Please use util/bin2comp.py to divide a (user, item) dataset into a training set and a test set, and extract pairwise comparisons from the training set. 

0. Prepare a dataset with (user, item) pairs. If the dataset consists of (user, item, ratings) triples, the numerical ratings are ignored. (Example: data/movielens1m.txt)

1. Run util/bin2comp.py to get training comparisons and test ratings. 
```
$ python util/bin2comp.py data/movielens1m.txt -o ml1m-bin -c 5000
```

2. Set the configuration options.
```
[input]
type = binary
train\_file = ml1m-bin\_train.dat
train\_file\_rating = ml1m-bin\_train\_ratings.lsvm 
test\_file = ml1m\_test-bin\_ratings.lsvm
```
(train_file_rating is not used for traning. It is for computing Precision@K where the training user-item pairs should be excluded)

3. Run the binary.
```
$ ./collrank
```
