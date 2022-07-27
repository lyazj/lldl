#include "Layer.h"
#include "Gradient.h"

#include <cstdlib>
#include <cassert>

void Layer::run_aff() const
{
  for(int j = 0; j < dim_out; ++j)
    data_aff[j] = 0;
  for(int i = 0; i < dim_in; ++i)
    if(data_in[i])
      for(int j = 0; j < dim_out; ++j)
        data_aff[j] += data_in[i] * W[i * dim_out + j];
}

float Layer::rand_wgt()
{
  return (rand() - RAND_MAX * 0.5f) / (RAND_MAX * 5.0f);
}

void Layer::init_wgt() const
{
  for(int i = 0; i < dim_in; ++i)
    for(int j = 0; j < dim_out; ++j)
      W[i * dim_out + j] = rand_wgt();
  for(int j = 0; j < dim_out; ++j)
    b[j] = rand_wgt();
}

void Layer::run_act() const
{
  act(dim_out, data_aff, data_out);
}

void Layer::alloc()
{
  W = new float[dim_in * dim_out];
  grad_W = new float[dim_in * dim_out];
  b = new float[dim_out];
  grad_b = new float[dim_out];
  grad_act = new float[dim_out * dim_out];
  data_aff = new float[dim_out];
  data_out = new float[dim_out];
}

Layer::Layer(int _dim_in, int _dim_out, act_t *_act)
{
  pred = nullptr;
  dim_in = _dim_in;
  data_in = new float[dim_in];
  succ = nullptr;
  dim_out = _dim_out;
  act = _act;

  alloc();
  init_wgt();
}

Layer::Layer(Layer *_pred, int _dim_out, act_t *_act)
{
  assert(!_pred->succ);
  _pred->succ = this;

  pred = _pred;
  dim_in = pred->dim_out;
  data_in = pred->data_out;
  succ = nullptr;
  dim_out = _dim_out;
  act = _act;

  alloc();
  init_wgt();
}

Layer::~Layer()
{
  delete[] W;
  delete[] grad_W;
  delete[] b;
  delete[] grad_b;
  delete[] grad_act;
  delete[] data_aff;

  if(pred)
    pred->succ = nullptr;
  else
    delete[] data_in;
  if(succ)
    succ->pred = nullptr;
  else
    delete[] data_out;
}

void Layer::calc_grad() const
{
  assert(succ);
  // grad(act, dim_out, data_aff, grad_act);
  for(int i = 0; i < dim_out; ++i)
  {
    float grad_bi = 0.0f;
    // for(int j = 0; j < dim_out; ++j)
if(data_aff[i] > 0)  // added
    for(int j = i; j < i + 1; ++j)
      for(int k = 0; k < succ->dim_out; ++k)
        // grad_bi += grad_act[i * dim_out + j]
        //   * succ->W[j * succ->dim_out + k]
        //   * succ->grad_b[k];
        grad_bi += succ->W[j * succ->dim_out + k]
          * succ->grad_b[k];
    grad_b[i] = grad_bi;
  }
  // for(int i = 0; i < dim_in; ++i)
  //   for(int j = 0; j < dim_out; ++j)
  //     grad_W[i * dim_out + j] = data_in[i] * grad_b[j];
}

void Layer::calc_grad(loss_t *loss, const float *data_exp) const
{
  assert(!succ);
  // grad(act, dim_out, data_aff, grad_act);
  float *grad_y_p = new float[dim_out];
  grad(loss, dim_out, data_exp, data_out, grad_y_p);
  for(int i = 0; i < dim_out; ++i)
  {
    float grad_bi = 0.0f;
    // for(int j = 0; j < dim_out; ++j)
if(data_aff[i] > 0)  // added
    for(int j = i; j < i + 1; ++j)
      // grad_bi += grad_act[i * dim_out + j] * grad_y_p[j];
      grad_bi += grad_y_p[j];
    grad_b[i] = grad_bi;
  }
  delete[] grad_y_p;
  // for(int i = 0; i < dim_in; ++i)
  //   for(int j = 0; j < dim_out; ++j)
  //     grad_W[i * dim_out + j] = data_in[i] * grad_b[j];
}

void Layer::forward() const
{
  assert(!pred);
  const Layer *layer = this;
  while(layer)
  {
    layer->run_aff();
    layer->run_act();
    layer = layer->succ;
  }
}

void Layer::backward(loss_t *loss,
    const float *data_exp, float learning_rate) const
{
  assert(!succ);
  calc_grad(loss, data_exp);
  const Layer *layer = this->pred;
  while(layer)
  {
    layer->calc_grad();
    layer = layer->pred;
  }
  layer = this;
  while(layer)
  {
    layer->optimize(learning_rate);
    layer = layer->pred;
  }
}

void Layer::optimize(float learning_rate) const
{
  // for(int i = 0; i < dim_in * dim_out; ++i)
  //   W[i] -= learning_rate * grad_W[i];
  for(int i = 0; i < dim_in; ++i)
    if(data_in[i])
      for(int j = 0; j < dim_out; ++j)
        W[i * dim_out + j] -= learning_rate * data_in[i] * grad_b[j];
  for(int i = 0; i < dim_out; ++i)
    b[i] -= learning_rate * b[i];
}
