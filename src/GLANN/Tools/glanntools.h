#ifndef GLANNTOOLS_H
#define GLANNTOOLS_H

#include "ofColor.h"

class GLANNTools
{
    public:
        GLANNTools();
        virtual ~GLANNTools();

        static ofFloatColor pack(double value);
        static double       unpack(ofFloatColor pixel);

    protected:
    private:
        static double fract(double f);
};

#endif // GLANNTOOLS_H
