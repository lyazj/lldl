#include "Mnist.h"

#include <cassert>

void Mnist::read_data(FILE *stream, float (*data)[28 * 28], int n)
{
  unsigned char buf[28 * 28];
  for(int i = 0; i < n; ++i)
  {
    int r = fread(buf, 1, 28 * 28, stream);
    assert(r == 28 * 28);
    for(int j = 0; j < 28 * 28; ++j)
      data[i][j] = buf[j] / 255.0f;
  }
}

void Mnist::read_label(FILE *stream, float (*label)[10], int n)
{
  unsigned char buf[n];
  int r = fread(buf, 1, n, stream);
  assert(r == n);
  for(int i = 0; i < n; ++i)
  {
    unsigned char val = buf[i];
    assert(val < 10);
    for(int j = 0; j < 10; ++j)
      label[i][j] = 0.0f;
    label[i][val] = 1.0f;
  }
}

Mnist::Mnist()
  : train_data(new float[60000][28 * 28]),
    train_label(new float[60000][10]),
    test_data(new float[10000][28 * 28]),
    test_label(new float[10000][10])
{
  // empty function body
}

Mnist::Mnist(const char *datafile) : Mnist()
{
  FILE *file = fopen(datafile, "rb");
  assert(file);
  read_data(file, (float (*)[28 * 28])train_data, 60000);
  read_label(file, (float (*)[10])train_label, 60000);
  read_data(file, (float (*)[28 * 28])test_data, 10000);
  read_label(file, (float (*)[10])test_label, 10000);
  fclose(file);
}

Mnist::~Mnist()
{
  delete[] train_data;
  delete[] train_label;
  delete[] test_data;
  delete[] test_label;
}
