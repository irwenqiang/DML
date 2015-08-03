#include "mpi.h"
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <iomanip>
#include <pthread.h>
#include "utils.cpp"
#include "opt_algo.cpp"


Utils u;
OPT_ALGO opt;

struct ThreadParam{
        std::vector<double>* theta;
        std::vector<std::vector<sparse_feature> >* fea_matrix;
        std::vector<double> *label;
        int proc_id;
        int n_proc;
};

void *opt_algo(void *arg){
    ThreadParam* args = (ThreadParam*) arg;
    std::cout<<args->proc_id<<std::endl;
    opt.owlqn(args->theta, args->fea_matrix, args->label, args->proc_id, args->n_proc);

}

int main(int argc,char* argv[]){  
    int myid, numprocs;
    MPI_Status status;
    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&myid);
    MPI_Comm_size(MPI_COMM_WORLD,&numprocs);
    opt.train_file = "./data/traindata.txt";
    opt.test_file = "./data/testdata.txt";
    opt.split_tag = '\t';
    u.mk_feature(opt.train_file, opt.split_tag, opt.fea_matrix, opt.label);
    u.get_fea_dim(opt.fea_matrix, opt.fea_dim);
    int fea_dim = opt.fea_dim;    
    std::cout<<fea_dim<<std::endl;
    MPI_Bcast(&fea_dim, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if(myid != 0) std::cout<<myid<<":"<<fea_dim<<std::endl;
    u.init_theta(opt.theta, opt.fea_dim);
    int n_threads = 3;
    std::vector<ThreadParam> params;
    std::vector<pthread_t> threads;
    for(int i = 0; i < n_threads; i++){
        ThreadParam param = {opt.theta, opt.fea_matrix, opt.label, myid, numprocs};
        params.push_back(param);
    } 
    /*
    for(int i = 0; i < params.size(); i++){
        pthread_t thread;
        int ret = pthread_create(&thread, NULL, &opt_algo, (void*)&(params[i])); 
        if(ret != 0) std::cout<<"process "<<i<<"failed(create thread faild.)"<<std::endl;
        else threads.push_back(thread);
            
    }
    for(int i = 0; i < threads.size(); i++){
        pthread_join(threads[i], 0); 
    }
    */
    MPI::Finalize();
    return 0;
}
