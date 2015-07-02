#include "mpi.h"
#include "opt_algo.h"
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <iomanip>

//OPT_ALGO opt;
OPT_ALGO::OPT_ALGO(){
}
OPT_ALGO::~OPT_ALGO(){
}
OPT_ALGO opt;


void predict(string test_file, vector<float>& theta, int myid){
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
        float y = opt.sigmoid(x);
        predict_result.push_back(y);
    }
    for(size_t j = 0; j < predict_result.size(); j++)
    {
        cout<<"predict rank %d result:"<<myid<<endl;
        cout<<predict_result[j]<<endl;
    }
}

int main(int argc,char* argv[]){  
    int myid, numprocs;
    MPI_Status status;
    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&myid);
    MPI_Comm_size(MPI_COMM_WORLD,&numprocs);
    
    string train_file = "traindata.txt";
    string test_file = "testdata.txt";

    mk_feature(train_file, opt.feature_matrix, opt.label, myid, numprocs);
    int feature_num = opt.feature_matrix.size();
    //MPI_Allreduce(&feature_num,);
    init_w(feature_num, opt.w);
    //opt.sgd(opt.w, myid, numprocs);
    opt.owlqn(opt.w, myid, numprocs);
    //opt.savemodel(opt.theta, myid);
    //opt.predict(test_file, opt.theta, myid);
    int ret_code = MPI_Finalize();
    fprintf(stderr,"%i,%i\n",myid, ret_code);
  return 0;
}
