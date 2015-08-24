#!/opt/apps/python/epd/7.3.2/bin/python

'''Convert binary click data to artificial comparison data'''
def pair_comp(x, y):
	if x[0] == y[0]:
		return x[1] - y[1]
	else:
		return x[0] - y[0]

def generate_comps(user_id, left_items, n_items, n_comps):
  right_items = [e for e in xrange(n_items) if e not in left_items]

  l = len(left_items)
  r = len(right_items)

  if not l or not r:
	  return	

  comps_list = []	
  _random, _int = random.random, int
  for i in xrange(int(n_comps)):
    li = _int(_random() * l)
    ri = _int(_random() * r)
    comps_list.append((left_items[li], right_items[ri]))

  comps_list.sort(cmp = pair_comp)
	
  for (l, r) in comps_list:
    print "%d %d %d" %(user_id, l + 1, r + 1)

def bin2comp(filename, n_comps):
  n_users = 0
  n_items = 0
  
  pairs_list = []
  with open(filename, 'r') as f:
    for line in f:
      (user_id, item_id) = line.strip().split()
      pairs_list.append((int(user_id)-1, int(item_id)-1))
      n_users = max(n_users, int(user_id))
      n_items = max(n_items, int(item_id))

  pairs_list.sort(cmp=pair_comp)

  idx = 0
  for u in xrange(n_users):
    left_items = []

    while pairs_list[idx][0] == u:
      left_items.append(pairs_list[idx][1])
      idx = idx + 1
      if idx == len(pairs_list):
        break

    if len(left_items) > 0 and len(left_items) < n_items:
      generate_comps(u+1, left_items, n_items, n_comps)

if __name__ == "__main__":
	import sys
	import random
	if (len(sys.argv)) < 3:
		sys.exit("Usage: " + sys.argv[0] + ' [click_data_file] [#comparisons_per_user]')
	
	bin2comp(sys.argv[1], sys.argv[2])
