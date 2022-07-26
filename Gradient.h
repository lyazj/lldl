#pragma once

typedef void act_t(int, const float *, float *);
typedef float loss_t(int, const float *, const float *);

void grad(act_t *act, int n, const float *y, float *g);
void grad(loss_t *loss, int n, const float *y_t, const float *y_p, float *g);
void grad_relu(int n, const float *y, float *g);
void grad_var_loss(int n, const float *y_t, const float *y_p, float *g);
