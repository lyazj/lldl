CXX = g++
CXXFLAGS = -O3 -Wall -Wshadow -Wextra

all: mnist.pickle mnist.bin LayerTest ModelTest

clean:
	$(RM) *.o mnist.pickle mnist.bin LayerTest ModelTest *.log

test-layer: LayerTest
	./$< 2>&1 | tee $<.log

test-model: ModelTest mnist.bin
	./$< 2>&1 | tee $<.log

mnist.pickle:
	./MnistToPickle.py

mnist.bin: mnist.pickle
	./PickleToBinary.py

%: %.o
	$(CXX) $(CXXFLAGS) $(filter %.o,$^) -o $@

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $< -o $@ -c

LayerTest: Layer.o Gradient.o Loss.o Activation.o
ModelTest: Model.o Mnist.o Layer.o Gradient.o Loss.o Activation.o

Activation.o: Activation.cpp Activation.h
Gradient.o: Gradient.cpp Gradient.h Activation.h Loss.h
Layer.o: Layer.cpp Layer.h Gradient.h
LayerTest.o: LayerTest.cpp Activation.h Layer.h
Loss.o: Loss.cpp Loss.h
Mnist.o: Mnist.cpp Mnist.h
Model.o: Model.cpp Model.h Layer.h
ModelTest.o: ModelTest.cpp Model.h Layer.h Loss.h Activation.h Mnist.h
