#pragma once

typedef float acc_t(int, const float *, const float *);
typedef float loss_t(int, const float *, const float *);

loss_t var_loss;
acc_t top_1_acc;
