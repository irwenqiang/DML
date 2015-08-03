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
        std::vector<double>* w;
        std::vector<std::vector<sparse_feature> >* fea_matrix;
        std::vector<double> *label;
        int proc_id;
        int n_proc;
};

void *opt_algo(void *arg){
    ThreadParam* args = (ThreadParam*) arg;
    std::cout<<args->proc_id<<std::endl;
    opt.owlqn(args->w, args->fea_matrix, args->label, args->proc_id, args->n_proc);

}

int main(int argc,char* argv[]){  
    int myid, numprocs;
    MPI_Status status;
    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&myid);
    MPI_Comm_size(MPI_COMM_WORLD,&numprocs);
    u.train_file = "./data/traindata.txt";
    u.test_file = "./data/testdata.txt";
    u.split_tag = '\t';
    u.mk_feature(u.train_file, u.split_tag);
    u.get_fea_dim();
    int fea_dim = u.fea_dim;    
    MPI_Bcast(&fea_dim, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if(myid != 0) std::cout<<myid<<":"<<fea_dim<<std::endl;
    u.init_w();
    //opt.sgd(opt.w, myid, numprocs);
    int n_threads = 3;
    std::vector<ThreadParam> params;
    for(int i = 0; i < n_threads; i++){
        ThreadParam param = {&u.w, &u.feature_matrix, &u.label, myid, numprocs};
        params.push_back(param);
    } 
    for(int i = 0; i < params.size(); i++){
        pthread_t thread;
        int ret = pthread_create(&thread, NULL, &opt_algo, (void*)&(params[i])); 
    }
    /*
    //opt.savemodel(opt.theta, myid);
    //opt.predict(test_file, opt.theta, myid);
    int ret_code = MPI_Finalize();
    fprintf(stderr,"%i,%i\n",myid, ret_code);*/
    MPI::Finalize();
    return 0;
}
