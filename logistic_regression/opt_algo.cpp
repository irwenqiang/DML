#include "opt_algo.h"


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
/*
void OPT_ALGO::sgd(vector<float>& w, int myid, int numprocs){
    size_t step = 0;
    string sample_line;
    vector<int> index;
    vector<float> delta_theta;
    cout<<"training start...!!!!..."<<endl;
    while(step < 2){
        int count = 0;
        size_t size = feature_matrix.size();
        for(size_t i = 0; i < size; i++){
            float index, value,x = 0.0, y = 0.0;
            for(size_t j = 0; j < feature_matrix[i].size(); j++){
                index = feature_matrix[i][j].idx;
                value = feature_matrix[i][j].val;
                x += w[index]*value;
            }
            float y_predict = sigmoid(x);
            for(size_t j = 0; j < feature_matrix[i].size(); j++)
            {
                delta_theta[index] += (y_predict-label[i])*value;
            }
            for(size_t j = 0; j < w.size(); j++){
                w[j] -= 0.1*delta_theta[j];
                delta_theta.clear();        
            }
        }
        step++;
    }
}
*/

//----------------------------owlqn--------------------------------------------
/*
float OPT_ALGO::fun(vector<float>& w){
    float f = 0.0;
    for(int i = 0; i < feature_matrix.size(); i++){
        double x = 0.0;
        for(int j = 0; j < feature_matrix[i].size(); j++){
            int id = feature_matrix[i][j].idx;
            int val = feature_matrix[i][j].val;
            x += w[id]*val;//maybe add bias later
        }
        double l = label[i]*log(1/sigmoid(-1*x)) + (1 - label[i])*log(1/sigmoid(x));
        f += l;
    }
    return f;
}

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
                                                      
void OPT_ALGO::sub_gradient(vector<float>& w,vector<float>& g, vector<float>& sub_g){
    if(c == 0){
        for(size_t j = 0; j < sub_g.size(); j++){
            sub_g[j] = -g[j];
        }
    }
    else{
        for(int j = 0; j < w.size(); j++){
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
void OPT_ALGO::two_loop(vector<float>& sub_g){
    //ro = 1/ ro;
    vector<float> q = sub_g;
    for(int loop = m; loop >= 0; loop--){
        double rolist[loop] = cblas_ddot(sub_g.size(), &ylist[loop], 1, &slist[loop], 1);
        alpha[loop] = cblas_ddot(sub_g.size(), slist[loop], 1, q, 1)/rolist[loop];
        addMult(q, ylist[loop], -alpha[loop]);
    }
    vector<float> lasty = ylist[m-1];
    double ydoty = dotProduct(lasty, lasty);
    double gamma = rolist[m-1]/ydoty;
    scale(sub_g, gamma);
    for(int loop = 0; loop < m; loop++){
        double beta = dotProduct(ylist[loop], sub_g)/rolist[loop];
        addMult(sub_g, slist[loop], -alpha[loop] - beta);
    }
}


void OPT_ALGO::fixdir(vector<float>& sub_g, vector<float>& g){
    for(size_t j = 0; j < dim; j++){
        if(sub_g[j] * g[j] >=0) g[j] = 0.0;
        else g[j] = sub_g[j];
    }
}
                     
void OPT_ALGO::linesearch(float old_f, vector<float>& w,
                          vector<float>&sub_g, vector<float>& g, vector<float>& nextw){
    float alpha = 1.0;
    double beta = 1e-4;
    double backoff = 0.5;
    while(true){
        for(int j = 0; j < w.size(); j++){
            nextw[j] = w[j] + alpha*sub_g[j];
        }
        fixdir(nextw, w);
        float newf = fun(nextw);
        if(newf <= old_f + beta* dot_(sub_g.size(), sub_g, 1, g, 1)){
            break;
        }
        alpha *= backoff;
        old_f = newf;
        for(int j = 0; j < w.size(); j++){
            w[j] = nextw[j];
        }
    }
}
*/  
/*
void OPT_ALGO::parallel_owlqn(float f, vector<float>& w, vector<float>& g){//inner thread, write g~ to g, mutex!
    //pthread_mutex_lock(&mutex);
    vector<float> next_w;
    vector<float> sub_g;
    float ro;
    ccopy_(sub_g.size(), sub_g, 1, g, 1);
    if(slist.size() >= m){
        two_loop(slist, ylist, rolist, sub_g);
    }
    fixdir(sub_g, g);
    double old_f = f;
    linesearch(old_f, w, sub_g, g, next_w);
    //pthread_mutex_unlock(&mutex);
    //update slist
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
    }
}*/

void OPT_ALGO::owlqn(int myid, int numprocs){
    int step = 0;
    std::string line;
    std::vector<int> index;
    std::cout<<"training by owlqn start "<<std::endl;
    //vector<float> sub_g(dim);
    //vector<float> g(dim);
    //vector<float> y_score(dim);
    //vector<float> pr_func;
    /*float oldval = 0.0;
    MPI_Allreduce(&w);
    f = fun(w);
    MPI_Allreduce(&w);
    grad(w, g);*/
    //while(step < 2){
    //    parallel_owlqn(f, w, g);        
    //}
}
















