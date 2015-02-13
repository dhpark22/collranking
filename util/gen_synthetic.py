#
# Generate training and test sets from a 1000 x 1000 random matrix with rank 10
#
import sys
import itertools
import random
import math
import numpy as np

def generate_synthetic(n_rows, n_cols, rank, n_train):
  print("Generating a random %d x %d matrix with rank %d" % (n_rows, n_cols, rank)) 
  train_comp   = open("../data/u%d_synth_train.dat" % n_train, "w")
  train_rating = open("../data/u%d_synth_train.lsvm" % n_train, "w")
  test_rating  = open("../data/u%d_synth_test.lsvm" % n_train, "w")  

  U = np.mat(np.random.rand(n_rows,rank)) / math.sqrt(rank)
  V = np.mat(np.random.rand(n_cols,rank)) / math.sqrt(rank)

  print "Now extracting %d training ratings per user" % n_train
  iidx = np.array(range(0, n_cols))

  for u in range(n_rows):
    X = U[u,:] * V.T 

    flag = np.array([0] * n_cols)
    random.shuffle(iidx)
    flag[iidx[0:n_train]] = 1

    for i in range(n_cols):
      if (flag[i]):
        train_rating.write("%d:%f " % (i+1, X[0,i]))
      else:
        test_rating.write("%d:%f " % (i+1, X[0,i]))
        
    train_rating.write("\n")
    test_rating.write("\n")

    pair = list(itertools.combinations(iidx[0:n_train],2))
    #random.shuffle(pair)
    for i in range(len(pair)):
      if (X[0,pair[i][0]] > X[0,pair[i][1]]):
        train_comp.write("%d %d %d\n" % (u+1, pair[i][0]+1, pair[i][1]+1))
      else:
        train_comp.write("%d %d %d\n" % (u+1, pair[i][1]+1, pair[i][0]+1))

  train_comp.close()
  train_rating.close()
  test_rating.close()    

if (len(sys.argv) != 2):
  print "Usage : python gen_synthetic.py [# training ratings per user]"
  sys.exit()

generate_synthetic(1000, 1000, 10, int(sys.argv[1]))

