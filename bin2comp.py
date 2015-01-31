#!/opt/apps/python/epd/7.3.2/bin/python

'''Convert binary click data to artificial comparison data'''
def pair_cmp(x, y):
	if x[0] == y[0]:
		return x[1] - y[1]
	else:
		return x[0] - y[0]

def bin2comp(filename, ntrain):
	nuser = 6040
	nitem = 3952

	ltable = [[] for i in xrange(nuser)]
	llen = []

	with open(filename, 'r') as f:
		for line in f:
			(u, e) = line.strip().split()
			ltable[int(u) - 1].append(int(e) - 1)

	o_lst = {}
	for u in xrange(nuser):
		l_lst = ltable[u]
		filt = set(l_lst)
		r_lst = [e for e in xrange(nitem) if e not in filt]
		lst = []

		l = len(l_lst)
		r = len(r_lst)

		if not l or not r:
			o_lst[u] = lst
			continue

		_random, _int = random.random, int
		for iter in xrange(int(ntrain) ):
			li = _int(_random() * l)
			ri = _int(_random() * r)
			lst.append((l_lst[li], r_lst[ri]) );			

		lst.sort(cmp = pair_cmp)
		o_lst[u] = lst

	for u in xrange(nuser):
		if not len(o_lst[u]):
			continue
		
		for (l, r) in o_lst[u]:
			print "%d %d %d" %(u + 1, l + 1, r + 1)


	'''
	ulist = []
	maxu = 0
	minu = 4000
	avg = 0
	for i in range(nuser):
		l = len(ltable[i])
		print l
		avg += l * 1.0 / nuser
		maxu = max(maxu, l)
		minu = min(minu, l)
		ulist.append(l)
	avg = int(avg)
	print "max items %d, min items %d, avg items %d" %(maxu, minu, avg)
	'''


if __name__ == "__main__":
	import sys
	import random
	if (len(sys.argv) ) < 3:
		sys.exit("Usage: " + sys.argv[0] + ' [click_data_file] [samples_per_user]')
	
	bin2comp(sys.argv[1], sys.argv[2])
