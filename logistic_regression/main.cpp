#include "mpi.h"
#include "opt_algo.h"
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <iomanip>

int main(int argc,char* argv[]){  
    int myid, numprocs;
    MPI_Status status;
    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&myid);
    MPI_Comm_size(MPI_COMM_WORLD,&numprocs);
    Utils u;
    //OPT_ALGO opt;
    u.train_file = "./data/traindata.txt";
    u.test_file = "./data/testdata.txt";
    u.mk_feature(u.train_file, u.split_tag);
    /*
    //int feature_num = opt.feature_matrix.size();
    //MPI_Allreduce(&feature_num,);
    init_w(feature_num, opt.w);
    //opt.sgd(opt.w, myid, numprocs);
    opt.owlqn(opt.w, myid, numprocs);
    //opt.savemodel(opt.theta, myid);
    //opt.predict(test_file, opt.theta, myid);
    int ret_code = MPI_Finalize();
    fprintf(stderr,"%i,%i\n",myid, ret_code);*/
    return 0;
}
