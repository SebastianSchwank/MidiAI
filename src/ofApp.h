#pragma once

#include "ofMain.h"

#include "ofUtils.h"

#include <vector>
#include <string>

#include "GLANN\ConvNet\conv.h"
#include "GLANN\RNN\rnn.h"

//This ofX extensions are required
#include "../../../../addons/ofxMaxim/src/ofxMaxim.h"
#include "../../../../addons/ofxMidi/src/ofxMidi.h"

class ofApp : public ofBaseApp{
	
	public:
		
		void setup();
		void update();
		void draw();
		
		void keyPressed(int key);
		void keyReleased(int key);
		void mouseMoved(int x, int y );
		void mouseDragged(int x, int y, int button);
		void mousePressed(int x, int y, int button);
		void mouseReleased(int x, int y, int button);
		void mouseEntered(int x, int y);
		void mouseExited(int x, int y);
		void windowResized(int w, int h);
		void dragEvent(ofDragInfo dragInfo);
		void gotMessage(ofMessage msg);
				
		void audioIn(float * input, int bufferSize, int nChannels); 
		void audioOut(float * output, int bufferSize, int nChannels);
		
		vector<float> norm(vector<float> signal);
		vector<float> processFftFrame(vector<float> signal);
		
		ofxMaxiFFT myFFT;
		//ofSoundStream soundStream

	//convNet			 mConv;
	rnnNet           mFFW0;
	rnnNet           mFFW1;
	
	int				probeLength			= 2048;
	int				midiNoteOutput		= 20;
	int				controllerOutput 	= 10;
	int				numSolutions		= 3;
	int 			midiOutput			= (midiNoteOutput*2+controllerOutput); //20MidiNote Outputs + 10 Midi Controller Outputs 
	int				outputNodes			= midiOutput*numSolutions;
	
	unsigned long	frameCounter = 0;
	
	float			learningRate = 0.0132;
	float			momentum	 = 0.00132;

	vector<float>	inputBufferTgt;
	vector<float>	inputBufferSyn;
	
	vector<float>		convOutputBuffer;
	vector<float>		midiOutputBuffer;
	
	float			maxVol;
	
	ofxMidiOut 		midiOut;
	
	int				bestSolution;
	float			bestSolutionErr;
	
	vector<float>	bestSolutionOverall;
	vector<float>	bestSolutionOverallNotes;
	float			bestSolutionOverallErr = 66666666;
	
	int				runtimeState = 0;
	int				solutionCounter = 1;
	int				channel;
	float			volThr = 0.001;
	
	bool			run = false;
	
	//ofxMaxiSample					targetSample;
	
};
