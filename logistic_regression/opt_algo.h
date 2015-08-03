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

    void owlqn(std::vector<double>* w, std::vector<std::vector<sparse_feature> >* fea_matrix, std::vector<double> *label, int myid, int num_procs);
    void sub_gradient(double g[], double sub_g[], int dim, double c);
    //void grad(vector<float>& w, vector<float>& g);
    void two_loop(int m, int dim, double *sub_g, double **s_list, double **y_list, double *ro_list, double *g);
    void parallel_owlqn(std::vector<double>* w, std::vector<std::vector<sparse_feature> >* fea_matrix, std::vector<double> *label);
    void fixdir(int dim, double *sub_g, double *g);
    double f_val(int dim, double *g, std::vector<std::vector<sparse_feature> >* fea_matrix, std::vector<double> *label);
    void linesearch(int dim, double old_f, double *sub_g, double *g, double *next_g, std::vector<std::vector<sparse_feature> >* fea_matrix, std::vector<double> *label);
private:
    //deque<vector<float> > ylist;
    //deque<vector<float> > slist;
    //vector<float> rolist;
    //vector<float> g, sub_g;
    //float f, nextw;
    //float c;
    //int dim, m;   
};
