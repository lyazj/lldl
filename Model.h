#pragma once

typedef void act_t(int, const float *, float *);
typedef float acc_t(int, const float *, const float *);
typedef float loss_t(int, const float *, const float *);
class Layer;

class Model {
private:
  int dim_in;  // 输入向量维度
  int dim_out;  // 输出向量维度
  float *data_in;  // 输入数据
  float *data_out;  // 输出数据
  Layer *head;  // 底层
  Layer *tail;  // 顶层
  acc_t *acc;  // 精度函数
  loss_t *loss;  // 损失函数
  float acc_value;  // 精度值
  float acc_sum;  // 精度值和
  float loss_value;  // 损失值
  float loss_sum;  // 损失值和
  float learning_rate;  // 学习率
  int nrun;  // 训练次数
  int ncnt;  // 计数次数

  void evaluate_run(const float *data_exp);
  void report_run() const;

public:
  static int report_per_runs;

  Model(int dim_in, acc_t *acc, loss_t *loss, float learning_rate);
  ~Model();
  int dim_in_value() const { return dim_in; }
  int dim_out_value() const { return dim_out; }
  float *data_in_addr() const { return data_in; }
  float *data_out_addr() const  { return data_out; }
  void add_layer(int dim_out, act_t *act);
  void finish_layer();
  void train(const float *data_in, const float *data_exp);
  void test(const float *data_in, const float *data_exp);
  void run(const float *data_in);
  int trained_times() const { return nrun; }
  float curr_acc() const { return acc_value; }
  float mean_acc() const { return acc_sum / ncnt; }
  float curr_loss() const { return loss_value; }
  float mean_loss() const { return loss_sum / ncnt; }
  void set_learning_rate(float learning_rate);
  void save(const char *modelfile) const;
  void load(const char *modelfile);
};
