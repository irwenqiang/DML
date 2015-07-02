vector<string> splitline(string &line，string split_tag){
    vector<string> tmp_vec;
    size_t beg = 0, end = 0;
    while((end = line.find_first_of(split_tag,beg)) != string::npos)
    {
        if(end > beg)
        {
            string index_str = line.substr(beg, end - beg);
            tmp_vec.push_back(index_str);
        }
        beg = end + 1;
    }
    if(beg < line.size())
    {
        string index_end = line.substr(beg);
        tmp_vec.push_back(index_end);
    }
    return tmp_vec;
}
int mk_feature(string train_file, vector<vector<sparse_feature> >& feature_matrix, vector<int>& label){
    cout<<"make feature start......"<<endl;
    ifstream fin(train_file.c_str());
    if(!fin)cerr<<"open error get feature number..."<<train_file<<endl;
    string line, splittag = ":";
    int max_index = 0;
    vector<string> feature_index;
    sparse_feature sf;
    vector<sparse_feature> key_val;
    
    int i = 0;
    while(getline(fin,line)){
        i++;
        key_val.clear();
        feature_index.clear();
        feature_index = splitline(line);
        int y = atoi(feature_index[0].c_str());
        label.push_back(y);
        int index = 0;
        for(int i = 1; i < feature_index.size(); i++){
            int beg = 0, end = 0;
            while((end = feature_index[i].find_first_of(":",beg)) != string::npos){
                if(end > beg){
                    string indexstr = feature_index[i].substr(beg,end-beg);
                    index = atoi(indexstr.c_str());
                    if (index > max_index)
                        max_index = index;
                    sf.idx = index - 1;
                    
                }
                //beg += 1; //this code must remain,it makes me crazy two days!!!
                beg = end + 1;
            }
            if(beg < feature_index[i].size()){
                string indexend = feature_index[i].substr(beg);
                int value = atoi(indexend.c_str());
                sf.val = value;
            }
            key_val.push_back(sf);
        }
        feature_matrix.push_back(key_val);
    }
    fin.close();
    return max_index;
    //cout<<"maxindex = "<<max_index<<endl;
}
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
