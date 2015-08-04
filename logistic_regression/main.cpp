#include "opt_algo.h"
#include "mpi.h"
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <iomanip>
#include <pthread.h>

struct ThreadParam{
    OPT_ALGO *opt;
    int proc_id;
    int n_procs;
};

void *opt_algo(void *arg){
   ThreadParam *args = (ThreadParam*)arg;
   args->opt->owlqn(args->opt, args->proc_id, args->n_procs); 
}

int main(int argc,char* argv[]){  
    int myid, numprocs;
    MPI_Status status;
    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&myid);
    MPI_Comm_size(MPI_COMM_WORLD,&numprocs);
    
    OPT_ALGO opt;

    std::string train_data_file = "./data/traindata.txt";
    std::string test_data_file = "./data/testdata.txt";
    std::string split_tag = "\t";
    opt.load_data(train_data_file, split_tag);
    std::cout<<opt.label.size()<<std::endl;
    opt.cal_fea_dim();
    MPI_Bcast(&opt.fea_dim, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if(myid != 0) std::cout<<myid<<":"<<opt.fea_dim<<std::endl;
    opt.init_theta();

    int n_threads = 3;
    std::vector<ThreadParam> params;
    std::vector<pthread_t> threads;
    for(int i = 0; i < n_threads; i++){
        ThreadParam param = {&opt, myid, numprocs};
        params.push_back(param);
    } 
    for(int i = 0; i < params.size(); i++){
        pthread_t thread;
        int ret = pthread_create(&thread, NULL, &opt_algo, (void*)&(params[i])); 
        if(ret != 0) std::cout<<"process "<<i<<"failed(create thread faild.)"<<std::endl;
        else threads.push_back(thread);
            
    }
    for(int i = 0; i < threads.size(); i++){
        pthread_join(threads[i], 0); 
    }
    MPI::Finalize();
    return 0;
}
