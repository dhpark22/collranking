#ifndef _COLLRANK_H
#define _COLLRANK_H

using namespace std;

struct rating
{
	int user_id;
	int item_id;
	int score;

	rating(): user_id(0), item_id(0), score(0) {}
	rating(int u, int i, int s): user_id(u), item_id(i), score(s) {}
	void setvalues(const int u, const int i, const int s) {
		user_id = u;
		item_id = i;
		score = s;
	}
	void swap(rating& r) {
		int temp;
		temp = user_id; user_id = r.user_id; r.user_id = temp;
		temp = item_id; item_id = r.item_id; r.item_id = temp;
		temp = score; score = r.score; r.score = temp;
	}
};

struct comparison
{
	int user_id;
	int item1_id;
	int item2_id;

	comparison(): user_id(0), item1_id(0), item2_id(0) {}
	comparison(int u, int i1, int i2): user_id(u), item1_id(i1), item2_id(i2) {}
	comparison(const comparison& c): user_id(c.user_id), item1_id(c.item1_id), item2_id(c.item2_id) {}
	void setvalues(const int u, const int i1, const int i2) {
		user_id = u;
		item1_id = i1;
		item2_id = i2;
	}
	void swap(comparison& c) {
		int temp;
		temp = user_id; user_id = c.user_id; c.user_id = temp;
		temp = item1_id; item1_id = c.item1_id; c.item1_id = temp;
		temp = item2_id; item2_id = c.item2_id; c.item2_id = temp;
	}
};

struct Node {
	int degree;
	map<int, int> neighbors;
	Node(): degree(0) {};
	int add_neighbor(int i) {
		if( neighbors.find(i) == neighbors.end() ) {
			neighbors.insert ( pair<int, int>(i, 1) );
			++degree;
			return 1;
		} else {
			++neighbors[i];
			return 0;
		}
	}
};

typedef struct rating rating;
typedef struct comparison comparison;
typedef struct Node Node;

// Graph storing the preference data
struct Graph {
	int m, n, E, omega, nparts;
	char* tf;
	vector<Node> nodes;
	vector<comparison> ucmp;		// pointer to array of comparisons sorted by users
	vector<comparison> pcmp;		// pointer to array of comparisons sorted by partitions
	vector<int> uidx;			// start/end indices of comparison data for each user
	vector<int> pidx;			// start/end indices of comparison data for each partition
	vector<vector<int> > p2idx;		// start/end indices of comparison data for each user in each partition
	map<int, int> buckets;

	Graph(): nparts(0) {}
	Graph(int np): nparts(np) {}

	void read_data(char* train_file) {
		tf = train_file;
		m = n = 0;
		ifstream f(train_file);
		if (f) {
			int u, i, j;
			while (f >> u >> i >> j) {
				n = max(u, n);
				m = max(i + 1, max(j + 1, m));
				ucmp.push_back(comparison(u - 1, i, j) );	// now user and item starts from 0
			}
			omega = ucmp.size();
		} else {
			printf("Error in opening the training file!\n");
			exit(EXIT_FAILURE);
		}
		f.close();

		uidx.resize(n + 1, 0);
		for (int i = 0; i < omega; ++i) {
			int u = ucmp[i].user_id;
			++uidx[u + 1];
		}
		for (int i = 1; i <= n; ++i) {
			uidx[i] += uidx[i - 1];
		}

		// construct the graph
		int offset = 1;
		nodes.resize(m + offset);		// skip node 0 to satisfy graclus format
		for (int i = 0; i < omega; ++i) {
			int j1 = ucmp[i].item1_id + offset;
			int j2 = ucmp[i].item2_id + offset;
			int n1 = nodes[j1].add_neighbor(j2);
			int n2 = nodes[j2].add_neighbor(j1);
			if (n1 + n2 == 1) {
				printf("j1=%d, j2=%d\n", j1, j2);
				exit(0);
			}
		}

		E = 0;		
		for (int i = offset; i < m + offset; ++i) {
			E += nodes[i].degree;
		}
		E /= 2;
	}

	void cluster() {
		// write the adjacent file
		int offset = 1;
		char gfile[20];
		strcpy(gfile, tf);
		strcat(gfile,".g");
		ofstream f(gfile);
		f << m << ' ' << E << " 1" <<endl;
		for (int i = offset; i < m + offset; ++i) {		// note: we skip node zero as the item and user numbering starts from zero
			for (map<int, int>::iterator it = nodes[i].neighbors.begin(); it != nodes[i].neighbors.end(); ++it) {
				f << it->first << ' ' << it->second << ' ';
			}
			f << endl;
		}
		f.close();
		
		// call gcluster
		char c_nparts[2];
		sprintf(c_nparts, "%d", nparts);
		// system call
		if (system(NULL) ) {
			char command[50]; 
			strcpy(command, "graclus -o ncut ");
			strcat(command, gfile);
			strcat(command, " ");
			strcat(command, c_nparts);
			printf("clustering command: %s\n", command);
			system(command);
		} else {
			printf("cannot execute graclus\n");
			exit(EXIT_FAILURE);
		}

		// read partition
		char partition[20];
		strcpy(partition, gfile);
		strcat(partition, ".part.");
		strcat(partition, c_nparts);
		printf("partition file: [%s]\n", partition);
		ifstream f2(partition);
		if (f2) {
			int i = 0;
			int j;
			while (f2 >> j) {
				buckets.insert(pair<int, int>(i, j) );
				++i;
			}		
		} else {
			printf("partition file not open!\n");
			exit(EXIT_FAILURE);
		}
		f2.close();

		// partition the file
		vector<vector<comparison> > tmp(nparts);
		for (int i = 0; i < omega; ++i) {
			comparison cur = ucmp[i];
			int j1 = cur.item1_id;
			int j2 = cur.item2_id;
			int g1 = buckets[j1];
			int g2 = buckets[j2];

			if(g1 == g2) {		// find edges belonging to the same group
				tmp[g1].push_back(cur);
			} else {		// assign abitrary
				if (rand() & 1) {
					tmp[g1].push_back(cur);
				} else {
					tmp[g2].push_back(cur);
				}
			}
		}

		// 1D indices for alt-SVM
		pidx.resize(nparts + 1);
		pidx[0] = 0;
		for (int i = 1; i <= nparts; ++i) {
			pidx[i] = pidx[i - 1] + tmp[i - 1].size();	
		}

		// 2D indices for SGD
		p2idx.resize(nparts);
		for (int i = 0; i < nparts; ++i) {
			p2idx[i].resize(n + 1, 0);
			for (int j = 0; j < tmp[i].size(); ++j) {
				comparison cur = tmp[i][j];
				int u = cur.user_id;
				++p2idx[i][u + 1];
			}
			for (int j = 1; j <= n; ++j) {
				p2idx[i][j] += p2idx[i][j - 1];
			}
		}

		pcmp.resize(pidx[nparts]);
		for (int i = 0; i < nparts; ++i) {
			for (int j = 0; j < tmp[i].size(); ++j) {
				pcmp[pidx[i] + j] = tmp[i][j];
			}
		}

		for (int i = 0; i < nparts; ++i) {
			printf("group %d, size %d\n", i, tmp[i].size() );
		}
	}
};

typedef struct Graph Graph;

//bool order_user(comparison a, comparison b) { return ((a.user_id < b.user_id) || ((a.user_id == b.user_id) && (min(a.item1_id, a.item2_id) < min(b.item1_id, b.item2_id)))); }
//bool order_item(comparison a, comparison b) { return (min(a.item1_id, a.item2_id) < min(b.item1_id, b.item2_id)); }
//bool order_user(rating a, rating b) { return ((a.user_id < b.user_id) || ((a.user_id == b.user_id) && (a.item_id < b.item_id))); }
//bool order_item(rating a, rating b) { return ((a.item_id < b.item_id) || ((a.item_id == b.item_id) && (a.user_id < b.user_id))); }

#endif
