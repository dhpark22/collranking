This is a collaborative work done by Jin Zhang and Dohyung Park.

extract_comps.cpp:		extract preference data from rating files of certain format 

collaborative_ranking.h:	header file for running collaborative ranking 

collaborative_ranking.cpp:	running collaborative ranking with alternative ranking SVM or Stochastic Gradient Descent (parallel version)

collaborative_ranking_seq.cpp:	running collaborative ranking with alternative ranking SVM (sequential version), matrix V are trained with all data once using liblinear

collaborative_ranking_seq2.cpp:	running collaborative ranking with alternative ranking SVM (sequential version), matrix V are trained with individual samples consecutively using liblinear

