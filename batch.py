import sys
import subprocess 

subprocess.call("./a.out $WORK/collrank_data/ml10m/u100_train_comps.dat $WORK/collrank_data/ml10m/u100_test_rating.lsvm 10 1000 16", shell=True) 
subprocess.call("./a.out $WORK/collrank_data/ml10m/u100_train_comps.dat $WORK/collrank_data/ml10m/u100_test_rating.lsvm 10 10000 16", shell=True) 

subprocess.call("./a.out $WORK/collrank_data/ml10m/u50_train_comps.dat $WORK/collrank_data/ml10m/u50_test_rating.lsvm 10 1000 16", shell=True) 
subprocess.call("./a.out $WORK/collrank_data/ml10m/u50_train_comps.dat $WORK/collrank_data/ml10m/u50_test_rating.lsvm 10 10000 16", shell=True) 

subprocess.call("./a.out $WORK/collrank_data/ml10m/u20_train_comps.dat $WORK/collrank_data/ml10m/u20_test_rating.lsvm 10 1000 16", shell=True) 
subprocess.call("./a.out $WORK/collrank_data/ml10m/u20_train_comps.dat $WORK/collrank_data/ml10m/u20_test_rating.lsvm 10 10000 16", shell=True) 

