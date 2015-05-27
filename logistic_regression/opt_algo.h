#include <string>
#include <fstream>
#include <vector>
#include <math.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>

using namespace std;
class OPT_ALGO{
    public:
        OPT_ALGO();
        ~OPT_ALGO();
        void sgd(string filename,
                   vector<float>& theta,
                   vector<float> &delta_theta,
                   vector<int>& label,
                   vec_vec& feature_matrix,
                   int myid,
                   int numprocs);

        void owlqn(string filename,
                   vector<float>& theta,
                   vector<float>& delta_theta,
                   vector<int>& label,
                   vec_vec& feature_matrix,
                   float c,
                   vector<float> two_loop_y,
                   vector<float> two_loop_s,
                   int m,
                   int myid,
                   int numprocs);
        float func(vector<vector<sparse_feature> >& feature_matrix, vector<float>& theta, vector<float>& y_predict);
        float cal_func(vector<float>& y_predict, vector<float>& theta, float c);
        void der_func(vector<float>& label, vector<float>& y_predict, int dim, vector<float>& g_f);
        void sub_gradient(vector<float>& g_f,vector<float> theta, vector<float>& sub_g);
        void two_loop(vector<vector<float> >& s, vector<vector<float> >& y, vector<float>& sub_g);
        void fixdir(vector<float>& sub_g, vector<float>& g_f);
        void linesearch(float oldval, vector<float> theta, vector<float> g, \
                        int step, float c, float alpha_step, float newval, vector<float> newtheta);
    private:
    string filename;
    deque<vector<float> > ylist;
    deque<vector<float> > slist;
    vector<float> rolist;
    vector<float> theta;//parameter of logistic regression
    vector<float> g, sub_g;
    float* f_val;
    vector<vector<sparse_feature> >& feature_matrix;
    vector<float> label;
    vector<float> y_score;
    int dim, m;
    
    
    
};
