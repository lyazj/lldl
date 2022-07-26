#include "Loss.h"

float var_loss(int n, const float *y_t, const float *y_p)
{
  float l = 0.0;
  for(int i = 0; i < n; ++i)
    l += (y_p[i] - y_t[i]) * (y_p[i] - y_t[i]);
  return l / n;
}

float top_1_acc(int n, const float *y_t, const float *y_p)
{
  int imax = 0;
  float ymax = y_p[0];
  for(int i = 1; i < n; ++i)
    if(ymax < y_p[i])
    {
      imax = i;
      ymax = y_p[i];
    }
  return y_t[imax];
}
