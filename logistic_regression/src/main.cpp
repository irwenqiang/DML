#include "opt_algo.h"
#include "mpi.h"
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <iomanip>
#include "gtest/gtest.h"

struct ThreadParam{
    OPT_ALGO *opt;
    int process_id;
    int n_process;
};

void *opt_algo(void *arg){
   ThreadParam *args = (ThreadParam*)arg;
   args->opt->owlqn(args->process_id, args->n_process); 
}

int main(int argc,char* argv[]){  
    int myid, numprocs;
    MPI_Status status;
    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&myid);
    MPI_Comm_size(MPI_COMM_WORLD,&numprocs);
    //exec by main thread    
    std::string train_data_file = "./data/train.txt";
    std::string test_data_file = "./data/test.txt";
    std::string split_tag = " ";
    
    OPT_ALGO opt;
    opt.fea_dim = 0;
    //get label and feature matrix
    opt.load_data(train_data_file, split_tag);
    int root = 0;
    MPI_Bcast(&opt.fea_dim, 1, MPI_INT, root, MPI_COMM_WORLD);
    std::cout<<opt.fea_dim<<std::endl;
    //std::cout<<"-----"<<std::endl;
    opt.init_theta();
    //pid_t mid;
    //mid = getpid();
    //std::cout<<mid<<std::endl;
    std::vector<ThreadParam> params;
    std::vector<pthread_t> threads;
    for(int i = 0; i < opt.n_threads; i++){//construct parameter
        ThreadParam param = {&opt, myid, numprocs};
        params.push_back(param);
    } 
    //multithread start
    for(int i = 0; i < params.size(); i++){
        pthread_t thread_id;
        std::cout<<thread_id<<std::endl;
        int ret = pthread_create(&thread_id, NULL, &opt_algo, (void*)&(params[i])); 
        if(ret != 0) std::cout<<"process "<<i<<"failed(create thread faild.)"<<std::endl;
        else threads.push_back(thread_id);
            
    }
    for(int i = 0; i < threads.size(); i++){//join threads function
        pthread_join(threads[i], 0); 
    }

    MPI::Finalize();
    return 0;
}
