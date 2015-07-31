#include <iostream>
#include <fstream>
#include "utils.h"

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
/*
void load_one_sample(string sample_filename)
{}

void init_w(int size, vector<float>& w){
    float init_w = 0.0;
    for(size_t j = 0; j < size; j++){
        w.push_back(init_w);
    }
}

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
void predict(string test_file, vector<float>& theta, int myid){
    cout<<"predic start-----------------------------------"<<endl;
    ifstream fin(test_file.c_str());
    string test_line;
    vector<float> predict_result;
    vector<string> predict_feature;
    float x;
    vector<int> preindex;
    vector<float> preval;
    while(getline(fin, test_line))
    {
        x = 0.0;
        predict_feature.clear();
        predict_feature = splitline(test_line);
        preindex.clear();
        preval.clear();
        for(size_t j = 0; j < predict_feature.size(); j++)
        {
            int beg = 0, end = 0;
            while((end = predict_feature[j].find_first_of(":",beg)) != string::npos)
            {
                if(end > beg)
                {
                    string string_sub = predict_feature[j].substr(beg, end - beg);
                    int k = atoi(string_sub.c_str());
                    preindex.push_back(k-1);
                }
                beg = end + 1;
            }
            string string_end = predict_feature[j].substr(beg);
            int t = atoi(string_end.c_str());
            preval.push_back(t);
            
        }
        for(size_t j = 0; j < preindex.size(); j++)
        {
            x += theta[preindex[j]]*preval[j];
        }
        float y = opt.sigmoid(x);
        predict_result.push_back(y);
    }
    for(size_t j = 0; j < predict_result.size(); j++)
    {
        cout<<"predict rank %d result:"<<myid<<endl;
        cout<<predict_result[j]<<endl;
    }
}
*/
