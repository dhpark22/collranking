Preference Completion: Large-scale Collaborative Ranking from Pairwise Comparison

This repo contains the implementation of the following algorithms:
- Alternating SVM (AltSVM)
- Stochastic Gradient Descent (SGD)
- Global Ranking from All-aggregated pairwise comparisons 

### Compilation
On a UNIX-based system with a C++11 supporting compiler and OpenMP API, Compile using the Makefile
```
$ make
```

### Experiments on numerical ratings
Our trained model can be tested in terms of NDCG@10 when the test set consists of numerical ratings. To compare with other rating based algorithms, we provide a code that extracts pairwise comparisons from ratings. 

0. Prepare a dataset with (user, item, ratings) triple (example - data/movielens1m.txt)
1. Run util/num2comp.py to get training comparisons and test ratings. 
```
$ python util/num2comp.py data/movielens1m.txt -o ml1m -n 50
```
2. Set the configuration options.
```
[input]
type = numeric
train\_file = ml1m\_train.dat
train\_file\_rating = ml1m\_train\_ratings.lsvm 
test\_file = ml1m\_test\_ratings.lsvm
```
3. Run ./collrank
```
$ ./collrank
```

### Experiments on binary ratings
Our trained model can also be tested in terms of Precision@K when the test set consists of binary ratings.

0. Prepare a dataset with (user, item) pairs. If the dataset consists of (user, item, ratings) triples, the numerical ratings are ignored (i.e., binarized) (example - data/movielens1m.txt)
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
3. Run ./collrank
```
$ ./collrank
```

