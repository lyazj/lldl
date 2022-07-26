#pragma once

#include <cstdio>

class Mnist {
private:
  Mnist();
  void read_data(FILE *stream, float (*)[28 * 28], int n);
  void read_label(FILE *stream, float (*)[10], int n);

public:
  const float (*const train_data)[28 * 28];
  const float (*const train_label)[10];
  const float (*const test_data)[28 * 28];
  const float (*const test_label)[10];

  Mnist(const char *datafile);
  ~Mnist();
};
