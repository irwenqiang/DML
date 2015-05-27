#include "lr.h"
#include <string.h>
#include <stdlib.h>

LR::LR()
{}
LR::~LR()
{}

float LR::sigmoid(float x)
{
    double sgm = 1/(1+exp(-(double)x));
    return (float)sgm;
}

vector<string> LR::splitline(string &line)
{
     vector<string> tmp_vec;
    size_t beg = 0, end = 0;
    string split_tag = "\t";
    while((end = line.find_first_of(split_tag,beg)) != string::npos)
    {
        if(end > beg)
        {
            string index_str = line.substr(beg, end - beg);
            tmp_vec.push_back(index_str);
        }
        beg = end + 1;
    }
    if(beg < line.size())
    {
    	string index_end = line.substr(beg);
	tmp_vec.push_back(index_end);
    }
    return tmp_vec;
}
int LR::get_feature_num(string sample_filename, vector<int>& label, vec_vec& feature_matrix)
{
    cout<<"Calculate feature number......"<<endl;
    ifstream fin(sample_filename.c_str());
    if(!fin)cerr<<"open error get feature number..."<<sample_filename<<endl;
    
    string line, splittag = ":";
    int max_index = 0;
    vector<string> feature_index;
    sparse_feature sf;
    vec key_val;
    while(getline(fin,line))
    {
        key_val.clear();
        feature_index.clear();
        feature_index = splitline(line);
        int y = atoi(feature_index[0].c_str());
        label.push_back(y);
        for(int i = 1; i < feature_index.size(); i++)
        {  
            int index = 0, beg = 0, end = 0;
            while((end = feature_index[i].find_first_of(":",beg)) != string::npos)
            {
                if(end > beg)
                {
                   string indexstr = feature_index[i].substr(beg,end-beg);
                    index = atoi(indexstr.c_str());
                    if (index > max_index)
                        max_index = index;
               //     cout<<index<<" ";
                   sf.id_index = index - 1;
                }
                beg += 1;
            }
            if(beg < feature_index[i].size())
            {
                string indexend = feature_index[i].substr(beg);
                int value = atoi(indexend.c_str());
                sf.id_val = value;
            }
            key_val.push_back(sf);
	    //if (index > max_index)
              //  max_index = index;
        }
        feature_matrix.push_back(key_val);

    }
    fin.close();
    cout<<"maxindex = "<<max_index<<endl;
    return max_index;
}
void load_one_sample(string sample_filename)
{}

void LR::init_theta(vector<float>& theta, int feature_size)
{
    float init_theta = 0.0;
    for(size_t i = 0; i < feature_size; i++)
	theta.push_back(init_theta);
}
void LR::train(string filename,vector<float>& theta, vector<int>& label,vector<vector<sparse_feature> >& feature_matrix)
{
    size_t step = 0;
    string sample_line;
    vector<float> delta_theta(theta.size(),0.0);
    vector<string> sample;
    vector<int> index;
    cout<<"traing start......"<<endl;
    while(step < 1)
    {
        ifstream fin(filename.c_str());
        int count = 0;
	    for(size_t i = 0; i < feature_matrix.size(); i++)
            {
                if(count % 10000 == 0)cout<<count<<endl;
                float val,x = 0.0, y = 0.0;
                for(size_t j = 0; j < feature_matrix[i].size(); j++)
                {
  		    int index = feature_matrix[i][j].id_index;
                    float val = feature_matrix[i][j].id_val; 
                    x += theta[index]*val;
                }
	        y = sigmoid(x);
	        for(size_t j = 0; j < feature_matrix[i].size(); j++)
	        {
                    delta_theta[feature_matrix[i][j].id_index] = y-label[i];
	        }
	        for(size_t j = 0; j < theta.size(); j++)
		    theta[j] -= 0.1*delta_theta[j];
                count++;
             }
	step++;
        cout<<"step "<<step<<endl;
	fin.close();
    }
}

void LR::predict(string trainfile, vector<float>& theta)
{
        cout<<"predic start-----------------------------------"<<endl;
	ifstream fin(trainfile.c_str());
	string train_line;
	vector<float> predict_result;
	vector<string> predict_feature;
    float x;
    vector<int> preindex;
    vector<float> preval;
    while(getline(fin, train_line))
    {
        x = 0.0;
        predict_feature.clear();
        predict_feature = splitline(train_line);
        preindex.clear();
        preval.clear();
        for(size_t j = 0; j < predict_feature.size(); j++)
        {
            int beg = 0, end = 0;
            while((end = predict_feature[j].find_first_of(":",beg)) != string::npos)
            {
                if(end > beg)
                {
                    string string_sub = predict_feature[j].substr(beg, end - beg);
                    int k = atoi(string_sub.c_str());
                    preindex.push_back(k-1);
                }
                beg += 1;
            }
                string string_end = predict_feature[j].substr(beg);
                int t = atoi(string_end.c_str());
                preval.push_back(t);
             
            //x += theta[k-1];
        }
        for(size_t j = 0; j < preindex.size(); j++)
        {
            //cout<<preindex[j]<<":"<<preval[j]<<endl;
            x += theta[preindex[j]]*preval[j];
        }
        float y = sigmoid(x);
        predict_result.push_back(y);
    }
    for(size_t j = 0; j < predict_result.size(); j++)
    {
        cout<<"predict result:"<<endl;
        cout<<predict_result[j]<<endl;
    }
}
struct A{ int a;char b; short c;char d;};
int main(int argc,char* argv[])
{  
    //cout<<sizeof(A)<<endl;
    LR lr;
    int feature_num = 0;
    string train_file = "train_feature";
    string predict_file = "predict_file";
    train_file = argv[1];
    feature_num = lr.get_feature_num(train_file, lr.label,lr.feature_matrix);
    /*for(size_t i = 0; i < lr.feature_matrix.size(); i++)
    {
        for(size_t j = 0; j < lr.feature_matrix[i].size(); j++)
            cout<<lr.feature_matrix[i][j].id_index<<":"<<lr.feature_matrix[i][j].id_val<<" ";
        cout<<endl;
    }*/
    lr.init_theta(lr.theta,feature_num);
    lr.train(train_file,lr.theta, lr.label, lr.feature_matrix);
    lr.label.clear();
    lr.feature_matrix.clear();
    lr.predict(predict_file, lr.theta);
    return 0;
}
