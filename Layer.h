#pragma once

typedef void act_t(int, const float *, float *);
typedef float loss_t(int, const float *, const float *);

class Layer {  // 深度层
private:
  int dim_in;  // 输入向量维度
  int dim_out;  // 输出向量维度
  act_t *act;  // 激活函数
  float *W;  // 权重
  float *b;  // 偏置
  float *grad_act;  // 激活函数梯度
  float *grad_W;  // 权重梯度
  float *grad_b;  // 偏置梯度
  float *data_in;  // 输入数据
  float *data_aff;  // 仿射结果
  float *data_out;  // 输出数据
  Layer *pred;  // 先驱层
  Layer *succ;  // 后继层

  void alloc();  // malloc() -> W, b, data_aff, data_out
  static float rand_wgt(); // -1e-1 -- 1e-1
  void init_wgt() const;  // rand_wgt() -> W, b
  void run_aff() const;  // data_in |--- (W, b) ---> data_aff
  void run_act() const;  // data_aff |--- act() ---> data_out
  void calc_grad() const;  // not for tail
  void calc_grad(  // for tail only
      loss_t *loss, const float *data_exp) const;
  void optimize(float learning_rate) const;  // SGD

public:
  Layer(int dim_in, int dim_out, act_t *act);
  Layer(Layer *pred, int dim_out, act_t *act);
  ~Layer();
  int dim_in_value() const;  // for head only
  int dim_out_value() const;  // for tail only
  float *data_in_addr() const;  // for head only
  float *data_out_addr() const;  // for tail only
  Layer *pred_layer() const { return pred; }
  Layer *succ_layer() const { return succ; }
  void forward() const;  // for head only
  void backward(loss_t *loss,  // for tail only
      const float *data_exp, float learning_rate) const;
};
