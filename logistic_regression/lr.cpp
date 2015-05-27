#include "mpi.h"
#include "lr.h"
#include "opt_algo.h"
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <iomanip>

LR::LR()
{}
LR::~LR()
{}

OPT_ALGO::OPT_ALGO(){}
OPT_ALGO::~OPT_ALGO(){}
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
void LR::mk_feature(string train_file, 
                   vector<int>& label, 
                   vec_vec& feature_matrix,
                   int myid,
                   int numprocs){
    cout<<"make feature start......"<<endl;
    ifstream fin(train_file.c_str());
    if(!fin)cerr<<"open error get feature number..."<<train_file<<endl;
    
    string line, splittag = ":";
    int max_index = 0;
    vector<string> feature_index;
    sparse_feature sf;
    vec key_val;
    int i = 0;
    while(getline(fin,line)){
        i++;
        key_val.clear();
        feature_index.clear();
        feature_index = splitline(line);
        int y = atoi(feature_index[0].c_str());
        label.push_back(y);
        int index = 0;
        for(int i = 1; i < feature_index.size(); i++){  
            int beg = 0, end = 0;
            while((end = feature_index[i].find_first_of(":",beg)) != string::npos){
                if(end > beg){
                    string indexstr = feature_index[i].substr(beg,end-beg);
                    index = atoi(indexstr.c_str());
                    if (index > max_index)
                        max_index = index;
                   sf.id_index = index - 1;
                   
                }
                //beg += 1; //this code must remain,it makes me crazy two days!!!
                beg = end + 1;
            }
            if(beg < feature_index[i].size()){
                string indexend = feature_index[i].substr(beg);
                int value = atoi(indexend.c_str());
                sf.id_val = value;
            }
            key_val.push_back(sf);
        }
        feature_matrix.push_back(key_val);
    }
    fin.close();
    //cout<<"maxindex = "<<max_index<<endl;
}
void load_one_sample(string sample_filename)
{}

void LR::init_theta(vector<float>& theta, vector<float> &delta_theta,int feature_size)
{
    float init_theta = 0.0;
    //cout<<feature_size<<endl;
    for(size_t i = 0; i < feature_size; i++){
	    theta.push_back(init_theta);
        delta_theta.push_back(init_theta);
    }
}

void LR::savemodel(vector<float> &theta,int myid){
    ofstream fout("train.model");
    fout.setf(ios::fixed,ios::floatfield);
    fout.precision(7);
    if(myid == 0)
	    for(int i = 0; i < 100; i++){
		    fout<<theta[i]<<endl;
	    }
    fout.close();
}

void LR::predict(string test_file, vector<float>& theta, int myid)
{
	cout<<"predic start-----------------------------------"<<endl;
	ifstream fin(test_file.c_str());
	string test_line;
	vector<float> predict_result;
	vector<string> predict_feature;
	float x;
	vector<int> preindex;
	vector<float> preval;
	while(getline(fin, test_line))
	{
		x = 0.0;
		predict_feature.clear();
		predict_feature = splitline(test_line);
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
                beg = end + 1;
            }
                string string_end = predict_feature[j].substr(beg);
                int t = atoi(string_end.c_str());
                preval.push_back(t);
             
        }
        for(size_t j = 0; j < preindex.size(); j++)
        {
            x += theta[preindex[j]]*preval[j];
        }
        float y = sigmoid(x);
        predict_result.push_back(y);
    }
    for(size_t j = 0; j < predict_result.size(); j++)
    {
        cout<<"predict rank %d result:"<<myid<<endl;
        cout<<predict_result[j]<<endl;
    }
}

int main(int argc,char* argv[])
{  
    int myid, numprocs;
    MPI_Status status;
    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&myid);
    MPI_Comm_size(MPI_COMM_WORLD,&numprocs);
    
    LR lr;
    OPT_ALGO opt;
    string train_file = "traindata.txt";
    string test_file = "testdata.txt";
    lr.mk_feature(train_file, 
                  lr.label,
                  lr.feature_matrix,
                  myid,
                  numprocs);
    int feature_num = lr.feature_matrix.size();
    lr.init_theta(lr.theta,
                  lr.delta_theta,
                  feature_num);
    opt.sgd(train_file,
             lr.theta, 
             lr.delta_theta,
             lr.label, 
             lr.feature_matrix,
             myid,
             numprocs);
    lr.savemodel(lr.theta, myid);
    lr.predict(test_file, lr.theta, myid);
    int ret_code = MPI_Finalize();
    fprintf(stderr,"%i,%i\n",myid, ret_code);
  return 0;
}
