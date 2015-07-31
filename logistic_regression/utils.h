#include <vector>
#include <string>

struct sparse_feature{
    int idx;
    double val;
};
class Utils{
public:
    std::vector<std::string> splitline(std::string&);
    int mk_feature(std::string file_name,  std::string);
private:
    std::vector<std::vector<sparse_feature> > feature_matrix;
    std::vector<double> label;
};
