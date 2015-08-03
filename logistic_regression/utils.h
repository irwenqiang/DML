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
    std::vector<std::string> split_line(std::string);
    void mk_feature(std::string, std::string, std::vector<std::vector<sparse_feature> >*, std::vector<double>*);
    void get_fea_dim(std::vector<std::vector<sparse_feature> >*, int);
    void init_theta(std::vector<double>*, int);

private:
    std::string line;
    std::string index_str;
    std::vector<std::string> feature_index;
    std::vector<sparse_feature> key_val;                                     
    sparse_feature sf;
};
#endif
