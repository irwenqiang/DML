#include "mpi.h"
//#include "opt_algo.h"
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <iomanip>
#include <pthread.h>
//#include "utils.h"
#include "utils.cpp"
#include "opt_algo.cpp"

int main(int argc,char* argv[]){  
    int myid, numprocs;
    MPI_Status status;
    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&myid);
    MPI_Comm_size(MPI_COMM_WORLD,&numprocs);
    Utils u;
    OPT_ALGO opt;
    u.train_file = "./data/traindata.txt";
    u.test_file = "./data/testdata.txt";
    u.split_tag = '\t';
    u.mk_feature(u.train_file, u.split_tag);
    u.get_fea_dim();
    int fea_dim = u.fea_dim;    
    MPI_Bcast(&fea_dim, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if(myid != 0) std::cout<<myid<<":"<<fea_dim<<std::endl;
    u.init_w();
    /*
    //opt.sgd(opt.w, myid, numprocs);
    opt.owlqn(opt.w, myid, numprocs);
    //opt.savemodel(opt.theta, myid);
    //opt.predict(test_file, opt.theta, myid);
    int ret_code = MPI_Finalize();
    fprintf(stderr,"%i,%i\n",myid, ret_code);*/
    MPI::Finalize();
    return 0;
}
