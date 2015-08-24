'''Generate training set and test set files from (user, item) pairs'''
from __future__ import print_function
import sys
import random
import argparse
import itertools

def pair_comp(x, y):
	if x[0] == y[0]:
		return x[1] - y[1]
	else:
		return x[0] - y[0]

def write_comps(f, user_id, left_items, n_items, n_comps):
  right_items = [e for e in xrange(n_items) if e not in left_items]

  n_left = len(left_items)
  n_right = len(right_items)

  if not n_left or not n_right:
	  return	

  comps_list = []	
  _random, _int = random.random, int
  for i in xrange(int(n_comps)):
    li = _int(_random() * n_left)
    ri = _int(_random() * n_right)
    comps_list.append((left_items[li], right_items[ri]))

  comps_list.sort(cmp = pair_comp)
	
  for (l, r) in comps_list:
    print(user_id, l, r, file=f)

def bin2comp(filename, output, f_train, f_test, n_comps):
  n_users = 0
  n_items = 0
  
  pairs_list = []
  f = open(filename, 'r')
  for line in f:
    tokens = line.strip().split()
    user_id = tokens[0]
    item_id = tokens[1]
    pairs_list.append((int(user_id), int(item_id)))
    n_users = max(n_users, int(user_id))
    n_items = max(n_items, int(item_id))
  f.close()

  print("Dataset for {0} users, {1} items loaded.".format(n_users, n_items))

  random.shuffle(pairs_list)
  
  n_train = int(float(len(pairs_list)) * f_train)
  train_pairs = pairs_list[:n_train]
  train_pairs.sort(cmp=pair_comp) 
 
  n_test  = min(int(float(len(pairs_list)) * f_test), len(pairs_list)-n_train) 
  test_pairs = pairs_list[n_train:(n_train+n_test)]
  test_pairs.sort(cmp=pair_comp)

  g1 = open(output+'_train.dat', 'w')
  idx = 0
  for u in xrange(1, n_users+1):
    left_items = []

    while train_pairs[idx][0] == u:
      left_items.append(train_pairs[idx][1])
      idx = idx + 1
      if idx == len(train_pairs):
        break

    if len(left_items) > 0 and len(left_items) < n_items:
      write_comps(g1, u, left_items, n_items, n_comps)
  g1.close()

  g2 = open(output+'_train_bin.dat','w')
  for uid, iid in train_pairs:
    print(uid, iid, file=g2)
  g2.close()

  g3 = open(output+'_test.dat','w')
  for uid, iid in test_pairs:
    print(uid, iid, file=g3)
  g3.close()
  

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('input_file',
                      help="Dataset with user-item-rating triples")
  parser.add_argument('-o', '--output_file', action='store', dest='output',
                      default="", help="Prefix for the output files")
  parser.add_argument('-n', '--train_frac', action='store', dest='f_train', type=float,
                      default=.9, help="Fraction of dataset for training (Default .9)") 
  parser.add_argument('-t', '--test_frac', action='store', dest='f_test', type=float,
                      default=.1, help="Fraction of dataset for test (Default .1)")
  parser.add_argument('-c', '--n_comps', action='store', type=int,
                      default=1000, help="Number of comparisons per user (Default 1000)")
  args = parser.parse_args()

  if (args.f_train + args.f_test > 1):
    raise Exception("F_TRAIN + F_TEST exceeds one!")

  if args.output == "":
    args.output = os.path.splitext(os.path.basename(args.input_file))[0]

  bin2comp(args.input_file, args.output, args.f_train, args.f_test, args.n_comps)
