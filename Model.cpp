#include "Model.h"
#include "Layer.h"

#include <cassert>
#include <iostream>

int Model::report_per_runs = 100;

void Model::evaluate_run(const float *data_exp)
{
  acc_value = acc(dim_out, data_exp, data_out);
  acc_sum += acc_value;
  loss_value = loss(dim_out, data_exp, data_out);
  loss_sum += loss_value;
  ++nrun;
  ++ncnt;
  if(ncnt == report_per_runs)
  {
    report_run();
    acc_sum = 0;
    loss_sum = 0;
    ncnt = 0;
  }
}

void Model::report_run() const
{
  std::clog << "Run " << trained_times() << ":"
            << "\tacc: " << curr_acc() << "\tmean_acc: " << mean_acc()
            << "\tloss: " << curr_loss() << "\tmean_loss: " << mean_loss()
            << std::endl;
}

Model::Model(int _dim_in, acc_t *_acc, loss_t *_loss, float _learning_rate)
{
  dim_in = _dim_in;
  dim_out = -1;
  data_in = nullptr;
  data_out = nullptr;
  head = nullptr;
  tail = nullptr;
  acc = _acc;
  loss = _loss;
  acc_value = 0.0f / 0.0f;
  acc_sum = 0.0f;
  loss_value = 0.0f / 0.0f;
  loss_sum = 0.0f;
  learning_rate = _learning_rate;
  nrun = 0;
  ncnt = 0;
}

Model::~Model()
{
  Layer *layer = tail;
  while(layer)
  {
    Layer *bye = layer;
    layer = layer->pred_layer();
    delete bye;
  }
}

void Model::add_layer(int _dim_out, act_t *act)
{
  if(tail)
    tail = new Layer(tail, _dim_out, act);
  else
  {
    head = tail = new Layer(dim_in, _dim_out, act);
    data_in = head->data_in_addr();
  }
}

void Model::finish_layer()
{
  dim_out = tail->dim_out_value();
  data_out = tail->data_out_addr();
}

void Model::train(const float *_data_in, const float *data_exp)
{
  test(_data_in, data_exp);
  tail->backward(loss, data_exp, learning_rate);
}

void Model::test(const float *_data_in, const float *data_exp)
{
  run(_data_in);
  evaluate_run(data_exp);
}

void Model::run(const float *_data_in)
{
  for(int i = 0; i < dim_in; ++i)
    data_in[i] = _data_in[i];
  head->forward();
}

void Model::set_learning_rate(float lr)
{
  learning_rate = lr;
}

void Model::save(const char *modelfile) const
{
  FILE *model = fopen(modelfile, "wb");
  if(!model)
  {
    std::cerr << "ERROR: cannot save model" << std::endl;
    return;
  }
  for(float i : {acc_value, acc_sum, loss_value, loss_sum, learning_rate})
  {
    int r = fwrite(&i, 1, 4, model);
    assert(r == sizeof(float));
  }
  for(int i : {nrun, ncnt})
  {
    int r = fwrite(&i, 1, 4, model);
    assert(r == sizeof(int));
  }
  for(Layer *layer = head; layer; layer = layer->succ_layer())
  {
    int _dim_in = layer->dim_in_value();
    int _dim_out = layer->dim_out_value();
    int _dim_io = _dim_in * _dim_out;
    for(int i : {_dim_in, _dim_out})
    {
      int r = fwrite(&i, 1, 4, model);
      assert(r == sizeof(int));
    }
    int r = fwrite(layer->weight(), 1, _dim_io * sizeof(float), model);
    assert(r == _dim_io * (int)sizeof(float));
    r = fwrite(layer->bias(), 1, _dim_out * sizeof(float), model);
    assert(r == _dim_out * (int)sizeof(float));
  }
  for(int i : {0, 0})
  {
    int r = fwrite(&i, 1, 4, model);
    assert(r == sizeof(int));
  }
  fclose(model);
}

void Model::load(const char *modelfile)
{
  FILE *model = fopen(modelfile, "rb");
  if(!model)
  {
    std::cerr << "ERROR: cannot load model" << std::endl;
    return;
  }
  for(float *p : {&acc_value, &acc_sum, &loss_value, &loss_sum, &learning_rate})
  {
    int r = fread(p, 1, 4, model);
    assert(r == sizeof(float));
  }
  for(int *p : {&nrun, &ncnt})
  {
    int r = fread(p, 1, 4, model);
    assert(r == sizeof(int));
  }
  for(Layer *layer = head; layer; layer = layer->succ_layer())
  {
    int _dim_in = layer->dim_in_value();
    int _dim_out = layer->dim_out_value();
    int _dim_io = _dim_in * _dim_out;
    for(int i : {_dim_in, _dim_out})
    {
      int val;
      int r = fread(&val, 1, 4, model);
      assert(r == sizeof(int));
      assert(i == val);
    }
    int r = fread(layer->weight(), 1, _dim_io * sizeof(float), model);
    assert(r == _dim_io * (int)sizeof(float));
    r = fread(layer->bias(), 1, _dim_out * sizeof(float), model);
    assert(r == _dim_out * (int)sizeof(float));
  }
  for(int i : {0, 0})
  {
    int val;
    int r = fread(&val, 1, 4, model);
    assert(r == sizeof(int));
    assert(i == val);
  }
  fclose(model);
}
