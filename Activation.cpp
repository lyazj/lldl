#include "Activation.h"

void relu(int n, const float *in, float *out)
{
  for(int i = 0; i < n; ++i)
    out[i] = in[i] > 0.0f ? in[i] : 0.0f;
}

// TODO
void softmax(int n, const float *in, float *out);
