#ifndef OPT_ALGO_H_
#define OPT_ALGO_H_

#include <string>
#include <fstream>
#include <vector>
#include <math.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>
#include <deque>
#include <pthread.h>

struct sparse_feature{
    int idx;
    double val;
};

class OPT_ALGO{
public:
    OPT_ALGO();
    ~OPT_ALGO();
    void load_data(std::string train_data_file, std::string split_tag);
    std::vector<std::string> split_line(std::string line, std::string split_tag);
    void cal_fea_dim();
    void init_theta();
    void owlqn(OPT_ALGO *opt, int proc_id, int n_procs);
    void parallel_owlqn(OPT_ALGO *opt, float *local_theta); 
    void f_grad(float *local_theta, float *g);
    float sigmoid(float x);
    void sub_gradient(float *g, float *sub_g);
    void two_loop(float *local_theta, float *sub_g, float **s_list, float **y_list, float *ro_list);
    void fix_dir(float *sub_g, float *g);
    void line_search(float *sub_g, float *local_theta, float *next_theta);
    float f_val(float *local_theta);

    std::vector<std::vector<sparse_feature> > fea_matrix;
    std::string index_str;
    std::vector<std::string> feature_index;
    std::vector<sparse_feature> key_val;                                     
    sparse_feature sf;

    std::vector<float> label;
    std::vector<float> theta;
    float c;
    int m;
    std::string train_file;
    std::string test_file;
    std::string split_tag;
    long int fea_dim;
private:
    
};
#endif
