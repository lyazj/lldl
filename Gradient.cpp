#include "Gradient.h"

#include <cassert>

#include "Activation.h"
#include "Loss.h"

// TODO
void grad(act_t *act, int n, const float *y, float *g)
{
  assert(act == relu);
  grad_relu(n, y, g);
}

// TODO
void grad(loss_t *loss, int n, const float *y_t, const float *y_p, float *g)
{
  assert(loss == var_loss);
  grad_var_loss(n, y_t, y_p, g);
}

void grad_relu(int n, const float *y, float *g)
{
  for(int i = 0; i < n; ++i)
    for(int j = 0; j < n; ++j)
      g[i * n + j] = i == j && y[i] > 0;
}

void grad_var_loss(int n, const float *y_t, const float *y_p, float *g)
{
  for(int i = 0; i < n; ++i)
    g[i] = 2 * (y_p[i] - y_t[i]) / n;
}
