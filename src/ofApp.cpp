#include "ofApp.h"
//Next Steps:
//Realtime wave target files
//Setting up controller-values with Keyboard
//--------------------------------------------------------------
void ofApp::setup(){	 
	
  //Set Environment Variables
    ofDisableArbTex();
    ofEnableAlphaBlending();
    //glEnable(GL_DEPTH_TEST);
    
    //ofSetFrameRate(24);
    ofSetVerticalSync(false);
	ofSoundStreamSetup(2, 2, 44100, probeLength/2, 2);
	ofSoundStreamListDevices();
	//ofSoundStreamStop();
	
	midiOut.openPort(5);
	midiOut.listPorts();
	
	myFFT.setup(probeLength/2, probeLength/2, probeLength/2);
	
	//targetSample.load(ofToDataPath("target.wav"));
	
	
	vector<float> testInput;
	for(int i = 0; i < probeLength/2; i++){
		testInput.push_back(0.5);
		inputBufferSyn.push_back(0.5);
		inputBufferTgt.push_back(0.5);
	}
	
	
	testInput = processFftFrame(inputBufferTgt);
	inputBufferSyn = processFftFrame(inputBufferSyn);
	inputBufferTgt = processFftFrame(inputBufferTgt);
	
	
	//mConv.initConv(16,testInput.size(),probeLength/10,learningRate,momentum);
	
	mFFW0.initRnn(0,testInput.size(),outputNodes,learningRate,momentum);
	mFFW1.initRnn(0,outputNodes,outputNodes,learningRate,momentum);
	
	maxVol  = 0.0;
	channel = 1;
	
}

vector<float> ofApp::processFftFrame(vector<float> signal){
	vector<float> result;
	int i = 0;
	while(!myFFT.process(signal[i]))i++;
	
	myFFT.magsToDB();
	
	for(int i = 0; i < signal.size()/2; i++)
		if(myFFT.magnitudes[i] > maxVol)
			maxVol = myFFT.magnitudesDB[i];
		
	for(int i = 0; i < signal.size()/2; i++)
		result.push_back(myFFT.magnitudesDB[i]/maxVol);
	
	return result;
}

vector<float> ofApp::norm(vector<float> signal){
	float max = 0.0;
	for(int i = 0; i < signal.size(); i++)
		if(signal[i] > max)
			max = signal[i];
	for(int i = 0; i < signal.size(); i++)
		signal[i] /= max;
	return signal;
}

//--------------------------------------------------------------
void ofApp::update(){
	
	//Switch on MIDI-Notes
	if(runtimeState == 2){
		
			//Wait till notes are turned off
			ofSleepMillis(100);
		
		for(int i = 0; i < midiNoteOutput; i++)
			midiOut.sendNoteOn(channel, (int)(127.0*midiOutputBuffer[midiOutput*(solutionCounter-1)+i*2]),  (int)(127.0*midiOutputBuffer[midiOutput*(solutionCounter-1)+i*2+1]));
		
		for(int i = 0; i < controllerOutput; i++)
			midiOut.sendControlChange(channel, i, (int)(127.0*midiOutputBuffer[midiOutput*(solutionCounter-1)+midiOutputBuffer.size()-i]));
			
			//Wait till notes are turned on
			ofSleepMillis(100);
			
		runtimeState = 3;
	}
	
	//cout << inputBufferTgt.size() << "\n";
	//cout << inputBufferSyn.size() << "\n";
}

//--------------------------------------------------------------
void ofApp::draw(){
	
	//Generate MIDI Notes
	if(runtimeState == 1){
		//convOutputBuffer = mConv.propergateFW(inputBufferTgt);
		
		vector<float> FFW = mFFW0.propergateFW(inputBufferTgt);
		midiOutputBuffer = mFFW1.propergateFW(FFW);
		runtimeState = 2;
		bestSolutionErr = 66666666;
		
		//ofSoundStreamStart();
	}
	
	//Associate the Output Midi Notes to the incoming Signal
	if(runtimeState == 4){
		//vector<float> convOutputBufferI = mConv.propergateFW(inputBufferSyn);
		//ofSoundStreamStop();
		
		vector<float> deviation;
		for(int i = 0; i < inputBufferSyn.size(); i++)
			deviation.push_back(inputBufferSyn[i]);
		
		vector<float> FFW = mFFW0.propergateFW(deviation);
		vector<float> midiOutputBufferI = mFFW1.propergateFW(FFW);
		
		vector<float> error;
		for(int i = 0; i < midiOutputBufferI.size(); i++)
			error.push_back(midiOutputBuffer[i%midiOutput+midiOutput*bestSolution] - midiOutputBufferI[i]);
		
		vector<float> bpErr = mFFW1.propergateBW(FFW,error);
		mFFW0.propergateBW(deviation,bpErr);
		//mConv.propergateBW(inputBufferSyn,bpErr);
		
		runtimeState = 1;
		
		//Train overall best Solution
		deviation.clear();
		for(int i = 0; i < bestSolutionOverall.size(); i++)
			deviation.push_back(bestSolutionOverall[i]);
		
		FFW = mFFW0.propergateFW(deviation);
		midiOutputBufferI = mFFW1.propergateFW(FFW);
		
		error.clear();
		for(int i = 0; i < midiOutputBufferI.size(); i++)
			error.push_back(bestSolutionOverallNotes[i%midiOutput] - midiOutputBufferI[i]);
		
		bpErr = mFFW1.propergateBW(FFW,error);
		mFFW0.propergateBW(deviation,bpErr);
		
		
	}
	
	
	// draw the audio Input:
	ofPushStyle();
		ofPushMatrix();
		ofTranslate(32, 150, 0);
			
		ofSetColor(225);
		ofDrawBitmapString("Audio Input Channel Synthesis", 4, 18);
		
		ofSetLineWidth(1);	
		ofDrawRectangle(0, 0, 900, 200);

		ofSetColor(58, 245, 135);
		ofSetLineWidth(2);
		
		ofNoFill();
					
			ofBeginShape();
			for (unsigned int i = 0; i < inputBufferSyn.size(); i++){
				float x =  ofMap(i, 0, inputBufferSyn.size(), 0, 900, true);
				ofVertex(x, 200 -inputBufferSyn[i]*180.0f);
			}
			ofEndShape(false);
			
		ofPopMatrix();
	ofPopStyle();
	
	
	ofPushStyle();
		ofPushMatrix();
		ofTranslate(32, 150, 0);
			
		ofSetColor(225);
		ofDrawBitmapString("Audio Input Channel Target Signal", 4, 18);

		ofSetColor(245, 58, 135);
		ofSetLineWidth(2);
		
		ofNoFill();
					
			ofBeginShape();
			for (unsigned int i = 0; i < inputBufferTgt.size(); i++){
				float x =  ofMap(i, 0, inputBufferTgt.size(), 0, 900, true);
				ofVertex(x, 200 -inputBufferTgt[i]*180.0f);
			}
			ofEndShape(false);
			
		ofPopMatrix();
	ofPopStyle();
	
	
	ofPushStyle();
		ofPushMatrix();
		ofTranslate(32, 150, 0);
			
		ofSetColor(225);
		ofDrawBitmapString("Audio Input Channel Target Signal", 4, 18);

		ofSetColor(0, 0, 255);
		ofSetLineWidth(2);
		
		ofNoFill();
					
			ofBeginShape();
			for (unsigned int i = 0; i < bestSolutionOverall.size(); i++){
				float x =  ofMap(i, 0, bestSolutionOverall.size(), 0, 900, true);
				ofVertex(x, 200 -bestSolutionOverall[i]*180.0f);
			}
			ofEndShape(false);
			
		ofPopMatrix();
	ofPopStyle();
	
	
	
	
	frameCounter++;
}

//--------------------------------------------------------------
void ofApp::audioIn(float * input, int bufferSize, int nChannels){
	
	
	vector<float> inTgt;
	for(int i = 0; i < bufferSize; i++)
		inTgt.push_back(0.5+0.5*input[i*nChannels+0]);
	
	
	inputBufferTgt.clear();
	inputBufferTgt = inTgt;
	
	inputBufferTgt = processFftFrame(inTgt);
	
	float max = 0.0;
	for(int i = 0; i < inputBufferTgt.size(); i++){
		if(inputBufferTgt[i] > max)
			max = inputBufferTgt[i];
	}
	for(int i = 0; i < inputBufferTgt.size(); i++)
		inputBufferTgt[i] /= max;
	
	
	
	if(runtimeState == 3){
		
		vector<float> inSyn;
		
		float vol = 0.0;
		
		for(int i = 0; i< bufferSize; i++){
			inSyn.push_back(0.5+0.5*input[i*nChannels+1]);
			vol += input[i*nChannels+1]*input[i*nChannels+1];
		}
		
		vol /= bufferSize;
		
		if(vol > volThr){
			vector<float> inputSyn = inSyn;
			
			inputSyn = processFftFrame(inSyn);
			float max = 0.0;
			for(int i = 0; i < inputSyn.size(); i++){
				if(inputSyn[i] > max)
					max = inputSyn[i];
			}
			for(int i = 0; i < inputSyn.size(); i++)
				inputSyn[i] /= max;
			
			float accErr = 0.0;
			for(int i = 0; i < inputSyn.size(); i++)
				accErr += (inputBufferTgt[i]-inputSyn[i])*(inputBufferTgt[i]-inputSyn[i]);
			
			if(accErr < bestSolutionErr){
				bestSolutionErr = accErr;
				bestSolution = solutionCounter - 1;
				inputBufferSyn = inputSyn;
				if(bestSolutionErr < bestSolutionOverallErr){
					bestSolutionOverall = inputSyn;
					bestSolutionOverallErr = bestSolutionErr;
					bestSolutionOverallNotes.clear();
					for(int i = 0; i < midiOutput; i++)
						bestSolutionOverallNotes.push_back(midiOutputBuffer[midiOutput*(solutionCounter-1)+i]);
				}
			}
			
			for(int i = 0; i < midiNoteOutput; i++)
				midiOut.sendNoteOff(channel, (int)(127.0*midiOutputBuffer[midiOutput*(solutionCounter-1)+i*2]),  (int)(127.0*midiOutputBuffer[midiOutput*(solutionCounter-1)+i*2+1]));
		
			if(solutionCounter % numSolutions == 0){
				runtimeState = 4;
				solutionCounter = 1;
			}
			else{
				solutionCounter++;
				runtimeState = 2;
			}
		}
		
	}
	
	//cout << inputBufferTgt.size();
	//cout << inputBufferSyn.size();
	
}

//-----------------------------------------------------------------
void ofApp::audioOut(float * output, int bufferSize, int nChannels){
	
	
}

//--------------------------------------------------------------
void ofApp::keyPressed  (int key){ 

	if(key == '1')midiOut.sendControlChange(channel, 0, 0);
	if(key == '2')midiOut.sendControlChange(channel, 1, 0);
	if(key == '3')midiOut.sendControlChange(channel, 2, 0);
	if(key == '4')midiOut.sendControlChange(channel, 3, 0);
	if(key == '5')midiOut.sendControlChange(channel, 4, 0);
	if(key == '6')midiOut.sendControlChange(channel, 5, 0);
	if(key == '7')midiOut.sendControlChange(channel, 6, 0);
	if(key == '8')midiOut.sendControlChange(channel, 7, 0);
	if(key == '9')midiOut.sendControlChange(channel, 8, 0);
	if(key == '0')midiOut.sendControlChange(channel, 9, 0);
	
	if(key == ' '){
		run = !run;
		if(run == false){
			runtimeState = 0;
			
		for(int i = 0; i < midiNoteOutput; i++)
			midiOut.sendNoteOn(channel, (int)(127.0*bestSolutionOverallNotes[i*2]),  (int)(127.0*bestSolutionOverallNotes[i*2+1]));
		
		for(int i = 0; i < controllerOutput; i++)
			midiOut.sendControlChange(channel, i, (int)(127.0*bestSolutionOverallNotes[bestSolutionOverallNotes.size()-i]));
		
		}else{
			runtimeState = 1;
		}
	}
	
}

//--------------------------------------------------------------
void ofApp::keyReleased(int key){ 
	
}

//--------------------------------------------------------------
void ofApp::mouseMoved(int x, int y ){
	
}

//--------------------------------------------------------------
void ofApp::mouseDragged(int x, int y, int button){
	
}

//--------------------------------------------------------------
void ofApp::mousePressed(int x, int y, int button){
	
}

//--------------------------------------------------------------
void ofApp::mouseReleased(int x, int y, int button){

}

//--------------------------------------------------------------
void ofApp::mouseEntered(int x, int y){

}

//--------------------------------------------------------------
void ofApp::mouseExited(int x, int y){

}

//--------------------------------------------------------------
void ofApp::windowResized(int w, int h){

}

//--------------------------------------------------------------
void ofApp::gotMessage(ofMessage msg){

}

//--------------------------------------------------------------
void ofApp::dragEvent(ofDragInfo dragInfo){ 

}

