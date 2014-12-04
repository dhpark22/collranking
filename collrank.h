#ifndef _COLLRANK_H
#define _COLLRANK_H

#include <algorithm>

using std::min;

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

typedef struct rating rating;
typedef struct comparison comparison;

bool order_user(comparison a, comparison b) { return ((a.user_id < b.user_id) || ((a.user_id == b.user_id) && (min(a.item1_id, a.item2_id) < min(b.item1_id, b.item2_id)))); }
bool order_item(comparison a, comparison b) { return (min(a.item1_id, a.item2_id) < min(b.item1_id, b.item2_id)); }

bool order_user(rating a, rating b) { return ((a.user_id < b.user_id) || ((a.user_id == b.user_id) && (a.item_id < b.item_id))); }
bool order_item(rating a, rating b) { return ((a.item_id < b.item_id) || ((a.item_id == b.item_id) && (a.user_id < b.user_id))); }

#endif
