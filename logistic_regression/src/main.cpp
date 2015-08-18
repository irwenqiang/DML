#include <cstring>
#include <cstdlib>
#include <ctime>
#include <iomanip>
#include <thread>
#include <atomic>
#include <cassert>
#include "gtest/gtest.h"
#include "include/opt_algo.h"
#include "mpi.h"

struct ThreadParam {
    OPT_ALGO *opt;
    pid_t main_thread_id;
    int process_id;
    int n_process;
};

/*
void *opt_algo(void* arg){
   ThreadParam *args = (ThreadParam*)arg;
   args->opt->owlqn(args->process_id, args->n_process); 
   return NULL;
}
*/
void optAlgo(ThreadParam* arg) {
	arg->opt->owlqn(arg->process_id, arg->n_process);
}

int main(int argc,char* argv[]){
    int myid, numprocs;
    // MPI_Status status;
    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&myid);
    MPI_Comm_size(MPI_COMM_WORLD,&numprocs);

    std::string train_data_file("./data/traindata.txt");
    std::string test_data_file("./data/testdata.txt");
    std::string split_tag = "\t";

    //call by main process
    OPT_ALGO opt(train_data_file, split_tag);
    MPI_Bcast(&opt.fea_dim, 1, MPI_INT, 0, MPI_COMM_WORLD);

    //multithread start
    std::vector<ThreadParam> params;
    // std::vector<pthread_t> threads;
	std::vector<std::thread> threads;
	
	ThreadParam param{&opt, myid, numprocs};
	for (int i = 0; i < opt.n_threads; ++i) {
		threads.push_back(std::thread(optAlgo, &param));
	}
	for (auto& u : threads) {
		u.join();
	}

    MPI::Finalize();
    return 0;
}
