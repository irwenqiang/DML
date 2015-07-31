#include <vector>
#include <string>

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
    std::string train_file;
    std::string test_file;
    std::string split_tag;

private:
    std::vector<std::vector<sparse_feature> > feature_matrix;
    std::string line;
    std::string index_str;
    std::vector<double> label;
    std::vector<std::string> feature_index;                                         
    std::vector<sparse_feature> key_val;                                     
    sparse_feature sf;
};
