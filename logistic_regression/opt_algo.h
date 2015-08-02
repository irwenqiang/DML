#include <string>
#include <fstream>
#include <vector>
#include <math.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>
#include <deque>
#include <pthread.h>
#include "utils.h"

class OPT_ALGO{
public:
    OPT_ALGO();
    ~OPT_ALGO();
    std::vector<float> w;//parameter of logistic regression
    
    //void load_one_sample(string sample_file);  
    float sigmoid(float x);
    //void sgd(int myid, int numprocs);

    void owlqn(std::vector<double>* w, std::vector<std::vector<sparse_feature> >* fea_matrix, int myid, int num_procs);
    void sub_gradient(std::vector<float>& w, std::vector<float>& g, std::vector<float>& sub_g);
    //float fun(vector<float>& w);
    //void grad(vector<float>& w, vector<float>& g);
    //void two_loop(vector<float>& sub_g);
    void parallel_owlqn(std::vector<double>* w, std::vector<std::vector<sparse_feature> >* fea_matrix);
    //void fixdir(vector<float>& sub_g, vector<float>& g);
    //void linesearch(float old_f, vector<float>&sub_g, vector<float>& g, vector<float>& nextw);
private:
    //deque<vector<float> > ylist;
    //deque<vector<float> > slist;
    //vector<float> rolist;
    //vector<float> g, sub_g;
    //float f, nextw;
    //float c;
    //int dim, m;   
};
