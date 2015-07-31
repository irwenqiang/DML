#include "mpi.h"
//#include "opt_algo.h"
//#include "utils.h"
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
    
    std::string train_file = "traindata.txt";
    std::string test_file = "testdata.txt";
    /*Utils::mk_feature(train_file, OPT_ALGO::feature_matrix, OPT_ALGO::label, myid, numprocs);
    int feature_num = opt.feature_matrix.size();
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
