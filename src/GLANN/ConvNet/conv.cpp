#include "conv.h"

convNet::convNet()
{
    //ctor
}

convNet::~convNet()
{
    //dtor
}

//Loades the shader and inits the fbos
bool convNet::initConv(int numKernels,int numInputs,int numOutputs,float learningRate,float momentum){
	kernels = new ANNData(numInputs/numOutputs,numKernels,learningRate,1.0,momentum);
	worker.initGLANN();
}

//This function propergates an input Vector through a given ANN returning
//it's output (in case of rekurrent networks do this multible times on the
//output Data)
vector<float> convNet::propergateFW(vector<float> input){
	vector<float> output;
	
	for(int i = 0; i < input.size()/kernels->getnumInputs(); i++){
		vector<float> inputSlice;
		for(int j = 0; j < kernels->getnumInputs(); j++){
			inputSlice.push_back(input[j+i*kernels->getnumInputs()]);
		}
		vector<float> outputSlice = worker.propergateFW(inputSlice,kernels);
		for(int i = 0; i < outputSlice.size(); i++)
			output.push_back(outputSlice[i]);
	}
	
    return output;
}

//This function trains the network with the given Data
//consisting of an input (already propergated by the net)
//it's output and the desired/expected values named as target
vector<float> convNet::propergateBW(vector<float> input, vector<float> error){
	vector<float> bpErr;
	for(int i = 0; i < input.size()/kernels->getnumInputs(); i++){
		vector<float> inputSlice;
		for(int j = 0; j < kernels->getnumInputs(); j++){
			inputSlice.push_back(input[j+i*kernels->getnumInputs()]);
		}
		vector<float> errorSlice;
		for(int j = 0; j < kernels->getnumOutputs(); j++){
			errorSlice.push_back(error[j+i*kernels->getnumOutputs()]);
		}
		vector<float> bpErrSlice = worker.propergateBW(inputSlice,errorSlice,kernels);
		for(int i = 0; i < bpErrSlice.size(); i++)
			bpErr.push_back(bpErrSlice[i]);
	}
	
    return bpErr;
}
