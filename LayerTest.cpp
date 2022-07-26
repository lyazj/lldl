#include "Activation.h"
#include "Layer.h"

int main()
{
  Layer A(28 * 28, 512, relu);
  Layer B(&A, 10, relu);
}
