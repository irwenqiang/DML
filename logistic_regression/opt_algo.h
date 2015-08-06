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
    float val;
};

class OPT_ALGO{
public:
    OPT_ALGO();
    ~OPT_ALGO();

    void load_data(std::string train_data_file, std::string split_tag);
    std::vector<std::string> split_line(std::string split_tag);
    void cal_fea_dim();
    void init_theta();
    void owlqn(int proc_id, int n_procs);
    void parallel_owlqn(); 
    void f_grad(float *para_w, float *para_g);
    float sigmoid(float x);
    void sub_gradient(float *local_g, float *local_sub_g);
    void two_loop(float *sub_g, float **s_list, float **y_list, float *ro_list, float *p);
    void fix_dir(float *w, float *next_w);
    void line_search(float *local_g);
    float f_val(float *w);
    //not shared by multithreads
    std::string line;
    std::vector<std::string> tmp_vec;
    std::string index_str;
    std::vector<std::string> feature_index;
    std::vector<sparse_feature> key_val;                                     
    sparse_feature sf;
    //shared by multithreads
    std::vector<std::vector<sparse_feature> > fea_matrix;
    std::vector<float> label;
    float *w;
    float *next_w;
    float *global_g;
    float *global_next_g;
    std::string train_file;
    std::string test_file;
    std::string split_tag;
    long int fea_dim;
    float c;
    int m;
    int n_threads;
    float global_old_loss_val;
    float global_new_loss_val;

private:
    
};
#endif
