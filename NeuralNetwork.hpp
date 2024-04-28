#include <eigen3/Eigen/Eigen>
#include <iostream>
#include <vector>
using namespace std;
typedef float Scalar;
typedef Eigen::MatrixXf Mat;
typedef Eigen::RowVectorXf RowVec;
typedef Eigne::VextorXf ColVec;

class NeuralNetwork{
    public:
        //Constructor
        NeuralNetwork(vector<uint> topology, Scalar alpha = Scalar(0.001));

        //Forward Propagation
        void ForwardProp(RowVec& input);

        //Backward Propagation
        void BackwardProp(RowVec& output);

        //Error Calculator
        void ErrorCal(RowVec& output);

        //Update weights
        void UpdateWeights();

        //Training
        void Train(vector<RowVec*> data);

        //Objeccts for neural network

        vector<RowVec*> neuronlayers; //Different layers of the network
        vector<RowVec*> deltas; // erorr contribution of each neuron
        vector<RowVec*> weights; // connection weights
        vector<RowVec*> cachelayers; // unactivated layers of the network
        Scalar alpha; //Learning Rate
        
};