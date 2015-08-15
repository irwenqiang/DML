/*void predict(string test_file, vector<float>& theta, int myid){
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
