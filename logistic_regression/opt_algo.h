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

    //call by main thread
    void load_data(std::string train_data_file, std::string split_tag);
    void cal_fea_dim();
    void init_theta();

    //call by threads 
    void owlqn(int proc_id, int n_procs);

    void f_grad(float *para_w, float *para_g);
    float sigmoid(float x);
    void sub_gradient(float *local_g, float *local_sub_g);
    void two_loop(float *sub_g, float **s_list, float **y_list, float *ro_list, float *p);
    void fix_dir(float *w, float *next_w);
    void line_search(float *local_g);
    float f_val(float *w);
    //shared by multithreads
    std::vector<std::vector<sparse_feature> > fea_matrix;//feature matrix shared by all threads
    std::vector<float> label;//label of instance shared by all threads
    float *w;//model paramter shared by all threads
    float *next_w;//model paramter after line search
    float *global_g;//gradient of loss function
    float *global_next_g;//gradient of loss function when arrive new w
    long int fea_dim;//feature dimension
    float c;
    int m;
    int n_threads;//thread number
    float global_old_loss_val;//loss value of loss function
    float global_new_loss_val;//loss value of loss function when arrive new w

private:
    std::vector<std::string> split_line(std::string split_tag); 
    void get_feature_struct(std::vector<std::string> feature_index);
    void parallel_owlqn();

    std::string line;
    std::vector<std::string> tmp_vec; 
    std::vector<std::string> feature_index;
    std::string index_str;
    sparse_feature sf;
    std::vector<sparse_feature> key_val;

};
#endif
