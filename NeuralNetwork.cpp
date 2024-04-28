using namespace std;
//Constructor
NeuralNetwork::NeuralNetwork(vector<uint> topology, Scalar learningRate): topology(topology), alpha(alpha)
{
    for (uint i = 0; i < topology.size(); i++) {
        // Initialize neuron layers
        uint neuronCount = (i == topology.size() - 1) ? topology[i] : topology[i] + 1;
        neuronlayers.push_back(new RowVec(neuronCount));
        cachelayers.push_back(new RowVec(neuronCount));
        deltas.push_back(new RowVec(neuronCount));
 
        // Initialize neuron and cache values
        if (i != topology.size() - 1) {
            neuronlayers.back()->coeffRef(topology[i]) = 1.0;
            cachelayers.back()->coeffRef(topology[i]) = 1.0;
        }
 
        // Initialize weights matrix
        if (i > 0) {
            uint inputNeuronCount = topology[i - 1] + 1;
            uint outputNeuronCount = (i != topology.size() - 1) ? topology[i] + 1 : topology[i];
            weights.push_back(new Matrix(inputNeuronCount, outputNeuronCount));
            weights.back()->setRandom();
 
            // Set bias weights to zero
            weights.back()->col(outputNeuronCount - 1).setZero();
 
            // Set identity connection for bias neuron
            weights.back()->coeffRef(inputNeuronCount - 1, outputNeuronCount - 1) = 1.0;
        }
    }
};

//Forward Propagation Function
void NeuralNetwork::ForwadProp(RowVec& input){
    //Assigning the input to the input layer
    neuronLayers.front()->block(0,0,1,neuronlayers.front()->size()-1);
    
    for(int i = 1; i < topology.size(); i++){
        //Forward Propagation
        (*neuronlayers[i]) = (*neuronlayers[i-1]) * (*weights[i-1]);

        //Applying activation function
        neuronlayers[i]->block(0,0,1,topology[i].unaryExpr([this](Scalar x) {return activationFuntion(x);}));
    }
}