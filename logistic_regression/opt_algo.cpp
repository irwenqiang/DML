#include <iostream>
#include <vector>
#include "opt_algo.h"

extern "C"{
#include <cblas.h>
}

OPT_ALGO::OPT_ALGO()
{
}

OPT_ALGO::~OPT_ALGO()
{
}

std::vector<std::string> OPT_ALGO::split_line(std::string line, std::string split_tag){
    std::vector<std::string> tmp_vec;
    size_t start = 0, end = 0;
    while((end = line.find_first_of(split_tag, start)) != std::string::npos){
        if(end > start){
            std::string index_str = line.substr(start, end - start);
            tmp_vec.push_back(index_str);
        }
        start = end + 1;
    }
    if(start < line.size()){
        std::string index_end = line.substr(start);
        tmp_vec.push_back(index_end);
    }
    return tmp_vec;
}


void OPT_ALGO::load_data(std::string data_file, std::string split_tag){
    std::ifstream fin(data_file.c_str(), std::ios::in);
    if(!fin) std::cerr<<"open error get feature number..."<<data_file<<std::endl;
    int y = 0, index = 0, value = 0;
    std::string line;
    while(getline(fin,line)){
        key_val.clear();
        feature_index.clear();
        feature_index = split_line(line, split_tag);
        y = atof(feature_index[0].c_str());
        label.push_back(y);
        for(int i = 1; i < feature_index.size(); i++){
            int start = 0, end = 0;
            while((end = feature_index[i].find_first_of(split_tag, start)) != std::string::npos){
                if(end > start){
                    index_str = feature_index[i].substr(start, end - start);
                    index = atoi(index_str.c_str());
                    sf.idx = index - 1;
                }
                //beg += 1; //this code must remain,it makes me crazy two days!!!
                start = end + 1;
            }
            if(start < feature_index[i].size()){
                std::string index_end = feature_index[i].substr(start);
                value = atoi(index_end.c_str());
                sf.val = value;
            }
            key_val.push_back(sf);
        }
        fea_matrix.push_back(key_val);
    }
    fin.close();
}

void OPT_ALGO::cal_fea_dim(){
    fea_dim = fea_matrix.size();
}

void OPT_ALGO::init_theta(){
    float init_theta = 0.0;
    c = 1.0;
    m = 6;
    for(int j = 0; j < fea_dim; j++){
        theta.push_back(init_theta);
    }
}


float OPT_ALGO::sigmoid(float x)
{
    float sgm = 1/(1+exp(-(float)x));
    return (float)sgm;
}
//----------------------------owlqn--------------------------------------------
float OPT_ALGO::f_val(float *local_theta){
    float f = 0.0;
    for(int i = 0; i < fea_matrix.size(); i++){
        float x = 0.0;
        for(int j = 0; j < fea_matrix[i].size(); j++){
            int id = fea_matrix[i][j].idx;
            float val = fea_matrix[i][j].val;
            x += *(local_theta + id) * val;//maybe add bias later
        }
        float l = label[i] * log(1/sigmoid(-1 * x)) + (1 - label[i]) * log(1/sigmoid(x));
        f += l;
    }
    return f;
}

void OPT_ALGO::f_grad(float *local_theta, float *g){
    float f = 0.0;
    for(int i = 0; i < fea_matrix.size(); i++){
        double x = 0.0, value;
        int index;
        for(int j = 0; j < fea_matrix[i].size(); j++){
            index = fea_matrix[i][j].idx;
            value = fea_matrix[i][j].val;
            x += *(local_theta + index) * value;
        }
        for(int j = 0; j < fea_matrix[i].size(); j++){
            *(g + j) += label[i] * sigmoid(x) * value + (1 - label[i]) * sigmoid(x) * value;
        }
    }
    for(int j = 0; j < fea_dim; j++){
        *(g + j) /= fea_matrix.size();
    }
}

void OPT_ALGO::sub_gradient(float *g, float *sub_g){
    if(c == 0.0){
        for(int j = 0; j < fea_dim; j++){
            *(sub_g + j) = -1 * *(g + j);
        }
    }
    else{
        for(int j = 0; j < fea_dim; j++){
            if(*(g + j) > 0){
                *(sub_g + j) = *(g + j) - c;
            }
            else if(*(g + j) < 0){
                *(sub_g + j) = *(g + j) - c;
            }
            else {
                if(*(g + j) - c > 0) *(sub_g + j) = c - *(g + j);
                else if(*(g + j) - c < 0) *(sub_g + j) = *(g + j) - c;
                else *(sub_g + j) = 0;
            }
        }
    }
}
void OPT_ALGO::two_loop(float *local_theta, float *sub_g, float **s_list, float **y_list, float *ro_list){
    float *q = new float[fea_dim];
    float *alpha = new float[m]; 
    cblas_dcopy(fea_dim, (double*)sub_g, 1, (double*)q, 1);
    for(int loop = m; loop >= 0; loop--){
        ro_list[loop] = cblas_ddot(fea_dim, (double*)(&(*y_list)[loop]), 1, (double*)(&(*s_list)[loop]), 1);
        alpha[loop] = cblas_ddot(fea_dim, (double*)(&(*s_list)[loop]), 1, (double*)q, 1)/ro_list[loop];
        cblas_daxpy(fea_dim, -1 * alpha[loop], (double*)(&(*y_list)[loop]), 1, (double*)q, 1);
    }
    float *last_y = new float[fea_dim];
    for(int j = 0; j < fea_dim; j++){
        last_y[j] = *((*y_list + m-1) + j);
    }
    float ydoty = cblas_ddot(fea_dim, (double*)last_y, 1, (double*)last_y, 1);
    float gamma = ro_list[m-1]/ydoty;
    float *p = new float[fea_dim];
    cblas_sscal(fea_dim, gamma,(float*)p, 1);
    for(int loop = 0; loop < m; loop++){
        float beta = cblas_ddot(fea_dim, (double*)(&(*y_list)[loop]), 1, (double*)p, 1)/ro_list[loop];
        cblas_daxpy(fea_dim, alpha[loop] - beta, (double*)(&(*s_list)[loop]), 1, (double*)p, 1);
    }
    for(int j = 0; j < fea_dim; j++){
        *(sub_g + j) = p[j];
    } 
}

void OPT_ALGO::fix_dir(float *sub_g, float *g){
    for(int j = 0; j < fea_dim; j++){
        if(*(sub_g + j) * *(g + j) >=0) *(g + j) = 0.0;
        else *(g + j) = *(sub_g + j);
    }
}

void OPT_ALGO::line_search(float *sub_g, float *local_theta, float *next_theta){
    float alpha = 1.0;
    float beta = 1e-4;
    float backoff = 0.5;
    while(true){
        float old_f_v = f_val(local_theta);
        for(int j = 0; j < fea_dim; j++){
            *(next_theta + j) = *(local_theta + j) + alpha * *(sub_g + j);
        }
        fix_dir(next_theta, local_theta);
        float new_f_v = f_val(local_theta);
        float *f_g = new float[fea_dim];
        f_grad(local_theta, f_g);
        if(new_f_v <= old_f_v + beta * cblas_ddot(fea_dim, (double*)f_g, 1, (double*)sub_g, 1)){
            break;
        }
        alpha *= backoff;
        for(int j = 0; j < fea_dim; j++){
            local_theta[j] = next_theta[j];
        }
        break;
    }
}

void OPT_ALGO::parallel_owlqn(OPT_ALGO *opt, float *local_theta){
    //pthread_mutex_lock(&mutex);
    std::cout<<fea_dim<<std::endl;
    float *sub_g = new float[fea_dim];
    float *g = new float[fea_dim];
    for(int j = 0; j < fea_dim; j++){
       local_theta[j] = opt->theta[j];
    }
    float *f_g_old = new float[fea_dim];
    f_grad(local_theta, f_g_old);
    int m = 6;
    float c = 1.0;
    float *ro_list = new float[fea_dim];
    int use_list_len = 0;
    float **s_list = new float*[m];
    s_list[0] = new float[m * fea_dim];
    for(int i = 1; i < m; i++){
        s_list[i] = s_list[i-1] + fea_dim; 
    }
    
    float **y_list = new float* [m];
    y_list[0] = new float[m * fea_dim];
    for(int i = 1; i < m; i++){
        y_list[i] = y_list[i-1] + fea_dim; 
    }
    f_grad(local_theta, g);
    sub_gradient(g, sub_g); 
    if(use_list_len >= m){
        two_loop(local_theta, sub_g, s_list, y_list, ro_list);
    }
    float *next_theta = new float[fea_dim];
    line_search(sub_g, local_theta, next_theta);
    //pthread_mutex_unlock(&mutex);
    //update slist
    cblas_daxpy(fea_dim, -1, (double*)local_theta, 1, (double*)next_theta, 1);
    cblas_dcopy(fea_dim, (double*)next_theta, 1, (double*)s_list[use_list_len], 1);
    //update ylist
    float *f_g_new = new float[fea_dim];
    f_grad(next_theta, f_g_new);
    cblas_daxpy(fea_dim, -1, (double*)f_g_old, 1, (double*)f_g_new, 1); 
    cblas_dcopy(fea_dim, (double*)f_g_new, 1, (double*)y_list[use_list_len], 1);

    use_list_len++;
    if(use_list_len > m){
        for(int j = 0; j < fea_dim; j++){
            *(*(s_list + (use_list_len - m) % m) + j) = 0.0;
            *(*(y_list + (use_list_len - m) % m) + j) = 0.0;        
        }
    }
}

void OPT_ALGO::owlqn(OPT_ALGO *opt, int proc_id, int n_procs){
    int step = 0;
    std::cout<<opt->fea_dim<<std::endl;
    float *local_theta = new float[fea_dim];
    while(step < 2){
        parallel_owlqn(opt, local_theta);        
        step++;
    }
}

