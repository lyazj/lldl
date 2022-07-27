#include "Model.h"
#include "Loss.h"
#include "Activation.h"
#include "Mnist.h"

#include <iostream>

int main()
{
  Model model(28 * 28, top_1_acc, var_loss, 0.02);
  model.add_layer(512, relu);
  model.add_layer(10, relu);
  model.finish_layer();

  Mnist mnist("mnist.bin");
  model.load("mnist.model");
  for(int e = 0; e < 1; ++e)
    for(int i = 0; i < 60000; ++i)
      model.train(mnist.train_data[i], mnist.train_label[i]);
  model.save("mnist.model");
  // for(int e = 0; e < 5; ++e)
  //   for(int i = 0; i < 60000; ++i)
  //     model.train(mnist.train_data[i], mnist.train_label[i]);
  // for(int e = 5; e < 10; ++e)
  // {
  //   model.set_learning_rate(0.1 / (e + 1));
  //   for(int i = 0; i < 60000; ++i)
  //     model.train(mnist.train_data[i], mnist.train_label[i]);
  // }

  float acc_sum = 0.0f;
  float loss_sum = 0.0f;
  for(int i = 0; i < 10000; ++i)
  {
    model.test(mnist.test_data[i], mnist.test_label[i]);
    acc_sum += model.curr_acc();
    loss_sum += model.curr_loss();
  }
  std::cout << "test result:"
            << "\tmean_acc: " << acc_sum / 10000
            << "\tmean_loss: " << loss_sum / 10000
            << std::endl;
}
