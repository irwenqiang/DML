#include "lr.h"
#include "opt_algo.h"
using namespace std;

LR lr;
void OPT_ALGO::sgd(vector<float> &delta_theta, int myid, int numprocs){
    size_t step = 0;
    string sample_line;
    vector<int> index;
    cout<<"training start...!!!!..."<<endl;
    while(step < 2){
        int count = 0;
        size_t size = feature_matrix.size();
        int start = myid * (size / numprocs);
        int stop = (myid+1) * (size / numprocs);
        if(myid == numprocs - 1) stop = size;
        for(size_t i = start; i < stop; i++){
            float val,x = 0.0, y = 0.0;
            for(size_t j = 0; j < feature_matrix[i].size(); j++){
                int index = feature_matrix[i][j].id_index;
                float val = feature_matrix[i][j].id_val;
                x += theta[index]*val;
            }
            float y_predict = lr.sigmoid(x);
            for(size_t j = 0; j < feature_matrix[i].size(); j++)
            {
                delta_theta[feature_matrix[i][j].id_index] +=
                (y_predict-label[i])*feature_matrix[i][j].id_val;
            }
            for(size_t j = 0; j < theta.size(); j++){
                theta[j] -= 0.1*delta_theta[j];
                delta_theta.clear();        
            }
        }
        step++;
    }
}
//----------------------------owlqn------------------------------------
float OPT_ALGO::val_func(vector<float>& w){
    float f = 0.0;
    for(int i = 0; i < feature_matrix.size(); i++){
        double x = 0.0;
        for(int j = 0; j < feature_matrix[i].size(); j++){
            int id = feature_matrix[i][j].id_index;
            int val = feature_matrix[i][j].id_val;
            x += w[id]*val;//maybe add bias later
        }
        double l = label[i]*log(1/sigmoid(-1*x)) + (1 - label[i])*log(1/sigmoid(x))
        f += l;
    }
    return f;
}

void OPT_ALGO::grad_func(vector<float>& w, vector<float>& g){
    float f = 0.0;
    for(int i = 0; i < feature_matrix.size(); i++){
        double x = 0.0;
        for(int j = 0; j < feature_matrix[i].size(); j++){
            int id = feature_matrix[i][j].id_index;
            int val = feature_matrix[i][j].id_val;
            x += w[id]*val;
        }
        for(int j = 0; j < feature_matrix[i].size(); j++){
            int val = feature_matrix[i][j].id_val;
            g[j] += label[i]*sigmoid(x)*val + (1-label[i])*sigmoid(x)*val;
        }
    }
    for(int j = 0; j < g.size(); j++){
        g[i] /= feature_matrix.size();
    }
}
                                                      
void OPT_ALGO::sub_gradient(vector<float>& g_f,vector<float> theta, vector<float>& sub_g){
    if(c == 0){
        for(size_t j = 0; j < sub_g.size(); j++){
            sub_g[j] = -g_f[j];
        }
    }
    else{
        for(int j = 0; j < theta.size(); j++){
            if(theta[j] > 0){
                sub_g[j] = g_f[j] - c;
            }
            else if(theta[j] < 0){
                sub_g[j] = g_f[j] - c;
            }
            else {
                if(g_f[j] - c > 0) sub_g[j] = c - g_f[j];
                else if(g_f[j] - c < 0) sub_g[j] = g_f[j] -c;
                else sub_g[j] = 0;
            }
        }
    }
}
void OPT_ALGO::two_loop(vector<float>& slist, vector<float>& ylist,
                        vector<float>& rolist, vector<float>& sub_g){
    //ro = 1/ ro;
    vector<float> q = sub_g;
    for(int loop = m; loop >= 0; loop--){
        float rolist[loop] = dotProduct(ylist[loop], slist[loop]);
        alpha[loop] = dotProduct(s[loop], q)/rolist[loop];
        addMult(q, ylist[loop], -alpha[loop]);
    }
    vector<float> lasty = ylist[m-1];
    double ydoty = dotProduct(lasty, lasty);
    double gamma = rolist[m-1]/ydoty;
    scale(sub_g, gamma)
    for(int loop = 0; loop < m; loop++){
        double beta = dotProduct(ylist[loop], p)/ro[loop];
        addMult(sub_g, s[loop], -alpha[loop] - beta);
}

                    
void OPT_ALGO::fixdir(vector<float>& sub_g, vector<float>& g){
    for(size_t j = 0; j < dim; j++){
        if(sub_g[j] * g_f[j] >=0) g_f = 0.0;
        else g_f[j] = sub_g[j];
    }
}
                     
void OPT_ALGO::linesearch(float old_val, vector<float>& w,
                          vector<float>&sub_g, vector<float>& nextw){
    float alpha = 1.0;
    double beta = 1e-4;
    double backoff = 0.5;
    while(true){
        for(int j = 0; j < w.size(); j++){
            nextw[j] = w[j] + alpha*sub_g[j];
        }
        fixdir(nextw, w);
        newval = val_func(nextw);
        if(newval <= oldval + beta* g*sub_g){
            break;
        }
        alpha *= backoff;
        oldval = newval;
        for(int j = 0; j < w.size(); j++){
            w[j] = nextw[j];
        }
}
                     
void OPT_ALGO::owlqn(int myid, int numprocs){
    int step = 0;
    dim = theta.size();
    string line;
    vector<int> index;
    cout<<"training by owlqn start "<<endl;
    vector<float> sub_g(dim);
    vector<float> g(dim);
    vector<float> y_score(dim);
    vector<float> pr_func;
                     
    float oldval = 0.0;
    newval = val_func(theta);
    grad_func(w, g);//g_f is deri of f
    while(step < 2){
        for(size_t i = start; i < stop; i++){
            scaleInto(sub_g, g, 1);
            if(slist.size() >= m){
                two_loop(slist, ylist, rolist, sub_g);
            }
            fixdir(sub_g, g);
            oldval = newval;
            linesearch(oldval, w, sub_g, nextw);
            //update slist
            scaleInto(nextw, w, -1)
            slist.push_back(nextw);
            //update ylist
            vector<float> next_g;
            grad_func(w, next_g);
            scaleInto(nextw_g, g, -1);
            ylist.push_back(next_g)
            //update rolist
            double ro = dotProduct(next_g, next_w);
            rolist.push_back(ro);
            if(s.size() > m){
                s.pop_front();
                y.pop_front();
            }
        }
    }
}
















