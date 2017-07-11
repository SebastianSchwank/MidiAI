#include "rnn.h"

rnnNet::rnnNet()
{
    //ctor
}

rnnNet::~rnnNet()
{
    //dtor
}

//Loades the shader and inits the fbos
bool rnnNet::initRnn(int numRekurrent,int numInputs,int numOutputs,float learningRate,float momentum){
	kernels = new ANNData(numInputs+numRekurrent,numOutputs,learningRate,1.0,momentum);
	for(int i = 0; i < numRekurrent; i++)
		rekurrent.push_back(0.0);
	worker.initGLANN();
}

//This function propergates an input Vector through a given ANN returning
//it's output (in case of rekurrent networks do this multible times on the
//output Data)
vector<float> rnnNet::propergateFW(vector<float> input){

	for(int i = 0; i < rekurrent.size(); i++)
		input.push_back(rekurrent[i]);

	rekurrent = worker.propergateFW(input,kernels);
	
    return rekurrent;
}

//This function trains the network with the given Data
//consisting of an input (already propergated by the net)
//it's output and the desired/expected values named as target
vector<float> rnnNet::propergateBW(vector<float> input, vector<float> error){
	vector<float> bpErr;

	for(int i = 0; i < rekurrent.size(); i++)
		input.push_back(rekurrent[i]);
	
	bpErr = worker.propergateBW(input,error,kernels);
	
	for(int i = 0; i < rekurrent.size(); i++)
		bpErr.pop_back();
	
    return bpErr;
}

void rnnNet::reset(){

	for(int i = 0; i < rekurrent.size(); i++)
		rekurrent[i] = 0.0;

}