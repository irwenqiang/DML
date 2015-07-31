#ifndef UTILS_H_
#define UTILS_H_

#include <vector>
#include <string>
#include <stdlib.h>

struct sparse_feature{
    int idx;
    double val;
};

class Utils{
public:
    Utils();
    ~Utils();
    std::vector<std::string> split_line();
    void mk_feature(std::string file_name, std::string split_tag);
    void get_fea_dim();
    void init_w();

    std::string train_file;
    std::string test_file;
    std::string split_tag;
    int fea_dim;
    std::vector<double> w;
    std::vector<std::vector<sparse_feature> > feature_matrix;
    std::vector<double> label;

private:
    std::string line;
    std::string index_str;
    std::vector<std::string> feature_index;                                         
    std::vector<sparse_feature> key_val;                                     
    sparse_feature sf;
};
#endif
