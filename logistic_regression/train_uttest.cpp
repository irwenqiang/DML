#include "opt_algo.h"
#include <vector>
#include "gtest/gtest.h"

class TrainTest: public testing::Test
{
    public: 
        //TrainTest() {}
        //virtual ~TrainTest(){}

        OPT_ALGO* opt;
        
        virtual void SetUp()
        {
            opt = new OPT_ALGO();
        }
        void TearDown()
        {
            delete opt;
        }
};

TEST_F(TrainTest, test_test)
{
    float a;
    a = 1.0;
    ASSERT_EQ(1.0, opt->sigmoid(a));
}

int main(int argc, char** argv){
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
    return 0;
}
