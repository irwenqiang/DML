#include "mpi.h"
#include "omp.h"
#include "lr.h"
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <iomanip>

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
int LR::get_feature_num(string sample_filename, vector<int>& label, vec_vec& feature_matrix,int myid,int numprocs)
{
    //cout<<"Calculate feature number......"<<endl;
    ifstream fin(sample_filename.c_str());
    if(!fin)cerr<<"open error get feature number..."<<sample_filename<<endl;
    
    string line, splittag = ":";
    int max_index = 0;
    vector<string> feature_index;
    sparse_feature sf;
    vec key_val;
    int i = 0;
    while(getline(fin,line)){
    //    if(i%10000 == 0)cout<<"for pthreadid: "<<myid<<"\t"<<i<<endl;
        i++;
        key_val.clear();
        feature_index.clear();
        feature_index = splitline(line);
        int y = atoi(feature_index[0].c_str());
        //cout<<y<<endl;
        label.push_back(y);
        for(int i = 1; i < feature_index.size(); i++){  
            int index = 0, beg = 0, end = 0;
            while((end = feature_index[i].find_first_of(":",beg)) != string::npos){
                if(end > beg){
                   string indexstr = feature_index[i].substr(beg,end-beg);
              //      cout<<indexstr<<"??i = "<<i<<endl;
                    index = atoi(indexstr.c_str());
                //    cout<<index<<endl;
                   // if(index == 0){cout<<i<<line<<endl;return 1;}
                    if (index > max_index)
                        max_index = index;
               //    n cout<<index<<" ";
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
            //cout<<"--------------------"<<endl;
	    //if (index > max_index)
              //  max_index = index;
        }
        feature_matrix.push_back(key_val);

    }
    fin.close();
    //int send = 1;
    //for(int tid = 1; tid < numprocs; tid++)
      //  MPI_Send(&send,1,MPI_INTEGER,tid,1,MPI_COMM_WORLD);
    cout<<"maxindex = "<<max_index<<endl;
    return max_index;
}
void load_one_sample(string sample_filename)
{}

void LR::init_theta(vector<float>& theta, vector<float> &delta_theta,int feature_size)
{
    float init_theta = 0.0;
    for(size_t i = 0; i < feature_size; i++){
	theta.push_back(init_theta);
        delta_theta.push_back(init_theta);
    }
}
void LR::train(string filename,vector<float>& theta, vector<float> &delta_theta, vector<int>& label,vector<vector<sparse_feature> >& feature_matrix,int myid,int numprocs)
{
    size_t step = 0;
    string sample_line;
    vector<int> index;
    //cout<<"traing start......"<<endl;
    while(step < 1)
    {
        int count = 0;
        size_t size = feature_matrix.size();
  	int start = myid * (size / numprocs);
        int stop = (myid+1) * (size / numprocs);
        if(myid == numprocs - 1) stop = size;
	    for(size_t i = start; i < stop; i++)
            {
               // if(i % 1000000 == 0)cout<<i<<endl;
                float val,x = 0.0, y = 0.0;
//#pragma omp parallel num_threads(16)
//{
			for(size_t j = 0; j < feature_matrix[i].size(); j++){
				int index = feature_matrix[i][j].id_index;
				float val = feature_matrix[i][j].id_val; 
				x += theta[index]*val;
			}
//}
		y = sigmoid(x);
//#pragma omp parallel num_threads(16)
//{
		for(size_t j = 0; j < feature_matrix[i].size(); j++)
		{
			delta_theta[feature_matrix[i][j].id_index] += y-label[i];
		}
//}
//#pragma omp parallel num_threads(16)
//{//
      //          cout<<i<<endl;
                if(i%10000 == 0){
		    for(size_t j = 0; j < theta.size(); j++)
			theta[j] -= 0.1*delta_theta[j]/1000;
                    delta_theta.clear();        
                }
//}
    //            cout<<"--------------------------------"<<endl;
	    }
	    step++;
	    //cout<<"step "<<step<<endl;
    }
}

void LR::savemodel(vector<float> &theta,int myid){
    ofstream fout("train.model");
    fout.setf(ios::fixed,ios::floatfield);
    fout.precision(7);
    //cout<<"model size = "<<theta.size()<<endl;
    //for(int i = 0; i < theta.size(); i++){
    int k = theta.size();
    cout<<"k = "<<k<<endl;
    if(myid == 0)
	    for(int i = 0; i < k; i++){
		    fout<<theta[i]<<endl;
	    }
    fout.close();
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
struct _feature{
        int id_index;
        float id_val;
};
int main(int argc,char* argv[])
{  

  /*  MPI_Init(&argc, &argv);
    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    int finalize_retcode = MPI_Finalize();
    if(0 == my_rank) fprintf(stderr, "Process, return_code\n");
    fprintf(stderr, "%i, %i\n", my_rank, finalize_retcode);*/
    //cout<<sizeof(A)<<endl;

    int myid, numprocs,ans = 0;
    MPI_Status status;
    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&myid);
    MPI_Comm_size(MPI_COMM_WORLD,&numprocs);
    clock_t time1 ,time2;
    LR lr;
    int feature_num = 0;
    //string train_file = "train_feature";
    string train_file = "sampling";
//    string predict_file = "predict_file";
//    train_file = argv[1];
/*    struct _feature matrix;
    MPI_Datatype myvalue;
    MPI_Datatype old_types[2];
    MPI_Aint indices[2];
    int blocklens[2];
    blocklens[0] = 1;
    blocklens[1] = 1;
    old_types[0] = MPI_INT;
    old_types[1] = MPI_FLOAT;
    MPI_Address(&matrix,&indices[0]);
    MPI_Address(&matrix,&indices[1]);
    indices[1] -= indices[0];
    indices[0] = 0;
    
    MPI_Type_struct(2,blocklens,indices,old_types,&myvalue);
    MPI_Type_commit(&myvalue);
    */
    time1 = clock();
    //if(myid == 0){
        feature_num = lr.get_feature_num(train_file, lr.label,lr.feature_matrix,myid,numprocs);

      //  int size = lr.feature_matrix.size();
       /* MPI_Datatype f_m;
        MPI_Type_vector(1,1,size,myvalue,&f_m);
        MPI_Type_commit(&f_m);
        
        MPI_Datatype f_matrix;
        MPI_Type_vector(size,1,size,myvalue,&f_matrix);
        MPI_Type_commit(&f_matrix);
        MPI_Bcast(&lr.feature_matrix,size*size,f_matrix,0,MPI_COMM_WORLD);
        ans = 1;
       */
        //for(int otherid = 1; otherid < numprocs; otherid++)
        //    MPI_Send(&ans,1,MPI_INT,otherid,1,MPI_COMM_WORLD);
    //}
     time2 = clock();
   /* for(size_t i = 0; i < 4; i++)
    {
        cout<<lr.feature_matrix[i].size()<<endl;
        for(size_t j = 0; j < lr.feature_matrix[i].size(); j++)
{
            cout<<lr.feature_matrix[i][j].id_index<<":"<<lr.feature_matrix[i][j].id_val<<" ";
         if(lr.feature_matrix[i][j].id_index == -1)
        {     
             cout<<i<<"-"<<j<<endl;
            return 1;
        } 
        cout<<endl;
 }
    }*/
    //cout<<"---------------"<<feature_num<<endl;
    lr.init_theta(lr.theta,lr.delta_theta,feature_num);
    //cout<< time2 - time1 <<endl;
    //if(myid != 0){
      //  MPI_Recv(&ans,1,MPI_INT,0,1,MPI_COMM_WORLD,&status);
        //if(ans == 1){
    //lr.delta_theta(lr.theta.size(),0.0);
            lr.train(train_file,lr.theta, lr.delta_theta,lr.label, lr.feature_matrix,myid,numprocs);
            lr.savemodel(lr.theta, myid);
       // }
    //}else{
      //  lr.train(train_file,lr.theta, lr.label, lr.feature_matrix,myid,numprocs);
    //}
        //lr.theta.clear();
        //lr.delta_theta.clear();
        //lr.feature_matrix.clear();
        clock_t time3 = clock();
        //cout<<time3 - time2<<endl;
    //lr.label.clear();
    //lr.feature_matrix.clear();
    //lr.predict(predict_file, lr.theta);
    int ret_code = MPI_Finalize();
    if(0 == myid) fprintf(stderr,"process,return_code");
    fprintf(stderr,"%i,%i",myid, ret_code);
  
  return 0;
}
