#include <iostream>
#include <vector>
#include "opt_algo.h"
#include "utils.h"

extern "C"{
#include <cblas.h>
}

OPT_ALGO::OPT_ALGO()
{
}

OPT_ALGO::~OPT_ALGO()
{
}

float OPT_ALGO::sigmoid(float x)
{
    double sgm = 1/(1+exp(-(double)x));
    return (float)sgm;
}
//----------------------------owlqn--------------------------------------------

double OPT_ALGO::f_val(int dim, double *g, std::vector<std::vector<sparse_feature> > *fea_matrix, std::vector<double> *label){
    double f = 0.0;
    for(int i = 0; i < (*fea_matrix).size(); i++){
        double x = 0.0;
        for(int j = 0; j < (*fea_matrix)[i].size(); j++){
            int id = (*fea_matrix)[i][j].idx;
            double val = (*fea_matrix)[i][j].val;
            x += *(g + id) * val;//maybe add bias later
        }
        double l = (*label)[i] * log(1/sigmoid(-1 * x)) + (1 - (*label)[i]) * log(1/sigmoid(x));
        f += l;
    }
    return f;
}
/*
void OPT_ALGO::grad(vector<float>& w, vector<float>& g){
    float f = 0.0;
    for(int i = 0; i < feature_matrix.size(); i++){
        double x = 0.0, value;
        int index;
        for(int j = 0; j < feature_matrix[i].size(); j++){
            index = feature_matrix[i][j].idx;
            value = feature_matrix[i][j].val;
            x += w[index]*value;
        }
        for(int j = 0; j < feature_matrix[i].size(); j++){
            g[j] += label[i]*sigmoid(x)*value + (1-label[i])*sigmoid(x)*value;
        }
    }
    for(int j = 0; j < g.size(); j++){
        g[j] /= feature_matrix.size();
    }
}
*/                                                      

void OPT_ALGO::sub_gradient(double g[], double sub_g[], int dim, double c){
    if(c == 0.0){
        for(size_t j = 0; j < dim; j++){
            sub_g[j] = -g[j];
        }
    }
    else{
        for(int j = 0; j < dim; j++){
            if(w[j] > 0){
                sub_g[j] = g[j] - c;
            }
            else if(w[j] < 0){
                sub_g[j] = g[j] - c;
            }
            else {
                if(g[j] - c > 0) sub_g[j] = c - g[j];
                else if(g[j] - c < 0) sub_g[j] = g[j] - c;
                else sub_g[j] = 0;
            }
        }
    }
}


void OPT_ALGO::two_loop(int m, int dim, double *sub_g, double **s_list, double **y_list, double *ro_list, double *g){
    //ro = 1/ ro;
    double q[dim];
    double alpha[m]; 
    cblas_dcopy(dim, sub_g, 1, q, 1);
    for(int loop = m; loop >= 0; loop--){
        ro_list[loop] = cblas_ddot(dim, &(*y_list)[loop], 1, &(*s_list)[loop], 1);
        alpha[loop] = cblas_ddot(dim, &(*s_list)[loop], 1, q, 1)/ro_list[loop];
        cblas_daxpy(dim, -1 * alpha[loop], &(*y_list)[loop], 1, q, 1);
    }
    double last_y[dim];
    for(int j = 0; j < dim; j++){
        last_y[j] = *((*y_list + m-1) + j);
    }
    double ydoty = cblas_ddot(dim, last_y, 1, last_y, 1);
    double gamma = ro_list[m-1]/ydoty;
    double p[dim];
    cblas_sscal(dim, gamma,(float*)p, 1);
    for(int loop = 0; loop < m; loop++){
        double beta = cblas_ddot(dim, &(*y_list)[loop], 1, p, 1)/ro_list[loop];
        cblas_daxpy(dim, alpha[loop] - beta, &(*s_list)[loop], 1, p, 1);
    }
    for(int j = 0; j < dim; j++){
        *(g + j) = p[j];
    } 
}


void OPT_ALGO::fixdir(int dim, double *sub_g, double *g){
    for(int j = 0; j < dim; j++){
        if(sub_g[j] * g[j] >=0) g[j] = 0.0;
        else g[j] = sub_g[j];
    }
}
 
void OPT_ALGO::linesearch(int dim, double old_f, double *sub_g, double *g, double *next_g, std::vector<std::vector<sparse_feature> >* fea_matrix, std::vector<double> *label){
    float alpha = 1.0;
    double beta = 1e-4;
    double backoff = 0.5;
    while(true){
        for(int j = 0; j < dim; j++){
            next_g[j] = g[j] + alpha*sub_g[j];
        }
        fixdir(dim, g, next_g);
        double new_f = f_val(dim, g, fea_matrix, label);
        /*
        if(new_f <= old_f + beta * cblas_ddot(dim, next_g, 1, g, 1)){
            break;
        }
        alpha *= backoff;
        old_f = newf;
        for(int j = 0; j < dim; j++){
            w[j] = next_g[j];
        }*/
        break;
    }
}
  

void OPT_ALGO::parallel_owlqn(std::vector<double>* w, std::vector<std::vector<sparse_feature> >* fea_matrix, std::vector<double>* label){//inner thread, write g~ to g, mutex!
    //pthread_mutex_lock(&mutex);
    int dim = (*w).size(); 
    //std::cout<<dim<<std::endl;
    //vector<double> next_w;
    double sub_g[dim];
    double g[dim];
    for(int j = 0; j < dim; j++){
       g[j] = (*w)[j];
       //std::cout<<g[j]<<std::endl;
    }
    int m = 6;
    double c = 1.0;
    double ro_list[] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    int one = 1;
    int use_slist_len = 0;
    int use_ylist_len = 0;
    double **s_list = new double*[m];
    s_list[0] = new double[m * dim];
    for(int i = 1; i < m; i++){
        s_list[i] = s_list[i-1] + dim; 
    }
    
    double **y_list = new double* [m];
    y_list[0] = new double[m * dim];
    for(int i = 1; i < m; i++){
        y_list[i] = y_list[i-1] + dim; 
    }
    sub_gradient(g, sub_g, dim, c); 
    if(use_slist_len >= m){
        two_loop(m, dim, sub_g, s_list, y_list, ro_list, g);
    }
    fixdir(dim, sub_g, g);
    double old_f = f_val(dim, g, fea_matrix, label);
    double next_g[dim];
    linesearch(dim, old_f, sub_g, g, next_g, fea_matrix, label);
    //pthread_mutex_unlock(&mutex);
    //update slist
    /*
    ccopy_(w.size(), nextw, 1, -w, 1);
    slist.push_back(next_w);
    //update ylist
    vector<float> next_g;
    grad(next_w, next_g);
    ccopy_(w.size(), next_g, 1, -g, 1);
    ylist.push_back(next_g);
    //update rolist
    ro = dot_(next_g, next_w);
    rolist.push_back(ro);
    if(slist.size() > m){
        slist.pop_front();
        ylist.pop_front();
    }*/
}

void OPT_ALGO::owlqn(std::vector<double>* w, std::vector<std::vector<sparse_feature> >* fea_matrix, std::vector<double> *label, int myid, int num_procs){
    int step = 0;
    //std::cout<<(*w).size()<<"------"<<myid<<std::endl;
    //std::vector<int> index;
    //std::cout<<"training by owlqn start "<<std::endl;
    //std::vector<double> sub_g(dim);
    //std::vector<float> g(dim);
    //vector<float> y_score(dim);
    //vector<float> pr_func;
    //float oldval = 0.0;
    //MPI_Allreduce(&w);
    //f = fun(w);
    //MPI_Allreduce(&w);
    //grad(w, g);
    while(step < 2){
        parallel_owlqn(w, fea_matrix, label);        
    }
}
















