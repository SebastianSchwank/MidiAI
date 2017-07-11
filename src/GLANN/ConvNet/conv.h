
#pragma once

#include "..\Worker\glann.h"
#include "..\DataModel\anndata.h"

#include <Vector>

class convNet
{
    public:
        convNet();
        virtual ~convNet();

        //Loades the shader and inits the fbos
        bool initConv(int numKernels,int numInputs,int numOutputs,float learningRate,float momentum);

        //This function propergates an input Vector through a given ANN returning
        //it's output (in case of rekurrent networks do this multible times on the
        //output Data)
        vector<float> propergateFW(vector<float> input);

        //This function trains the network with the given Data
        //consisting of an input (already propergated by the net)
        //it's output and the desired/expected values named as target
        vector<float> propergateBW(vector<float> input,
                                   vector<float> error);

		ANNData* kernels;

    protected:
    private:
		GLANN	 worker; 
		
};

