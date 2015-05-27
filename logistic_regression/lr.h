#include <string>
#include <fstream>
#include <vector>
#include <math.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>
using namespace std;

struct sparse_feature{
	int id_index;
 	float id_val;  
};
typedef vector<sparse_feature> vec; 
typedef vector<vector<sparse_feature> > vec_vec;
class LR{
    public:
	    LR();
  	    ~LR();
  	    vector<float> theta;
        vector<float> delta_theta;
	    vec sample_feature_vec;
        vec_vec feature_matrix;
        vector<int> label;
    	void load_one_sample(string sample_file);  
        float sigmoid(float x);
        vector<string> splitline(string &line); 
	    void init_theta(vector<float>& theta, vector<float>& delta_theta,int feature_size);
        void mk_feature(string sample_file, vector<int>& label,vec_vec& feature_matrix,int myid,int numprocs);
        void savemodel(vector<float> &theta, int myid);
        void predict(string filename, vector<float>& theta, int myid);
    private:
	      
};
