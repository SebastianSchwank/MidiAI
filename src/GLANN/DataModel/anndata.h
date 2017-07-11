#ifndef ANNDATA_H
#define ANNDATA_H

#include "ofImage.h"
#include "ofFbo.h"

#include "glanntools.h"

class ANNData
{
    public:
        ANNData(int inputs,int outputs, float learningRate, float steepness, float momentum);
        virtual ~ANNData();
		
		void resetNet();

        float getSteepness();
        float getLearningRate();
        float getMomentum();

        int getnumInputs();
        int getnumOutputs();
		
        ofImage mInput;
        ofImage mOutput;
        ofImage mError;
	
        ofImage mWeights;
        ofImage mMomentum;
		
        ofFbo *mFbo;

    private:
		
        float learningRate;

        float steepness;
        float momentum;

        int inputs,outputs;
};

#endif // ANNDATA_H
