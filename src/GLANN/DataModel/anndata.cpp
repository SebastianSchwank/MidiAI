#include "anndata.h"

ANNData::ANNData(int inputs,int outputs, float learningRate, float steepness, float momentum)
{
    this->inputs = inputs+1;
    this->outputs = outputs;
    this->learningRate = learningRate;
    this->steepness = steepness;
    this->momentum = momentum;

    //Init the Weights to small!! random Values
    mWeights.allocate(inputs, outputs, OF_IMAGE_COLOR_ALPHA);
    for(int x = 0; x < inputs; x++){
        for(int y= 0; y < outputs; y++){
            ofFloatColor pixelColor = GLANNTools::pack(ofRandom(0.3,0.7));
            mWeights.setColor(x,y,pixelColor);
        }
    }
    mWeights.update();
    //mWeights.reloadTexture();

    //Init the Momentum to "zero" = 0.5
    mMomentum.allocate(inputs,outputs,OF_IMAGE_COLOR_ALPHA);
    for(int x = 0; x < inputs; x++){
        for(int y= 0; y < outputs; y++){
            ofFloatColor pixelColor = GLANNTools::pack(0.5);
            mMomentum.setColor(x,y,pixelColor);
        }
    }
    mMomentum.update();
    //mMomentum.reloadTexture();

    mFbo = new ofFbo();
    mFbo->allocate(inputs,outputs,GL_RGBA);

    mInput.allocate(inputs,1,OF_IMAGE_COLOR_ALPHA);
    mError.allocate(1,outputs,OF_IMAGE_COLOR_ALPHA);
    mOutput.allocate(1,outputs,OF_IMAGE_COLOR_ALPHA);

}

void ANNData::resetNet(){
	//Init the Weights to small!! random Values
    for(int x = 0; x < inputs; x++){
        for(int y= 0; y < outputs; y++){
            ofFloatColor pixelColor = GLANNTools::pack(ofRandom(0.48,0.52));
            mWeights.setColor(x,y,pixelColor);
        }
    }
    mWeights.update();
    //mWeights.reloadTexture();

    //Init the Momentum to "zero" = 0.5
    for(int x = 0; x < inputs; x++){
        for(int y= 0; y < outputs; y++){
            ofFloatColor pixelColor = GLANNTools::pack(0.5);
            mMomentum.setColor(x,y,pixelColor);
        }
    }
    mMomentum.update();
}

float ANNData::getSteepness(){
    return steepness;
}

float ANNData::getLearningRate(){
    return learningRate;
}

float ANNData::getMomentum(){
    return momentum;
}

int ANNData::getnumInputs(){
    return inputs;
}

int ANNData::getnumOutputs(){
    return outputs;
}

ANNData::~ANNData()
{
    //dtor
}