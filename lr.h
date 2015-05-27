#include <string>
#include <fstream>
#include <vector>
#include <math.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>
using namespace std;

struct sparse_feature{
	vector<int> feature_id_vec;
 	vector<float> feature_value_vec;  
};
class LR{
    public:
	LR();
  	~LR();
  	vector<float> theta;
	vector<sparse_feature> sample_feature_vec;
        vector<int> label;
	void load_one_sample(string sample_file);  
        float sigmoid(float x);
        vector<string> splitline(string &line); 
	void init_theta(vector<float>& theta,int feature_size);
        int get_feature_num(string sample_file);
        void train(string filename,vector<float>& theta);
        void predict(string filename, vector<float>& theta);
    private:
	      
};
