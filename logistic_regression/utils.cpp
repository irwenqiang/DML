#include <iostream>
#include <fstream>
#include "utils.h"

Utils::Utils()
{
}

Utils::~Utils()
{
}

std::vector<std::string> Utils::split_line(){
    std::vector<std::string> tmp_vec;
    size_t start = 0, end = 0;
    while((end = line.find_first_of(split_tag, start)) != std::string::npos){
        if(end > start){
            std::string index_str = line.substr(start, end - start);
            tmp_vec.push_back(index_str);
        }
        start = end + 1;
    }
    if(start < line.size()){
        std::string index_end = line.substr(start);
        tmp_vec.push_back(index_end);
    }
    return tmp_vec;
}

void Utils::mk_feature(std::string file_name, std::string split_tag){
    std::ifstream fin(file_name.c_str(), std::ios::in);
    if(!fin) std::cerr<<"open error get feature number..."<<file_name<<std::endl;
    int y = 0, index = 0, value = 0;
    while(getline(fin,line)){
        key_val.clear();
        feature_index.clear();
        feature_index = split_line();
        y = atoi(feature_index[0].c_str());
        label.push_back(y);
        for(int i = 1; i < feature_index.size(); i++){
            int start = 0, end = 0;
            while((end = feature_index[i].find_first_of(split_tag, start)) != std::string::npos){
                if(end > start){
                    index_str = feature_index[i].substr(start, end - start);
                    index = atoi(index_str.c_str());
                    sf.idx = index - 1;
                }
                //beg += 1; //this code must remain,it makes me crazy two days!!!
                start = end + 1;
            }
            if(start < feature_index[i].size()){
                std::string index_end = feature_index[i].substr(start);
                value = atoi(index_end.c_str());
                sf.val = value;
            }
            key_val.push_back(sf);
        }
        feature_matrix.push_back(key_val);
    }
    fin.close();
}
  
void Utils::get_fea_dim(){
    fea_dim = feature_matrix.size();
}
//void load_one_sample(string sample_filename)
//{}

void Utils::init_w(){
    float init_w = 0.0;
    for(size_t j = 0; j < fea_dim; j++){
        w.push_back(init_w);
    }
}
/*
void savemodel(vector<float> &theta,int myid){
    ofstream fout("train.model");
    fout.setf(ios::fixed,ios::floatfield);
    fout.precision(7);
    if(myid == 0)
        for(int i = 0; i < 100; i++){
            fout<<theta[i]<<endl;
        }
    fout.close();
}
*/
