
CXX = g++

CXXFLAGS = -std=c++11 -O2

SRCS = main.cpp radiator_cpu.cpp

TARGET = radiator_cpu

all: $(TARGET)

$(TARGET):
	$(CXX) $(CXXFLAGS) $(SRCS) -o $(TARGET)

clean:
	rm -f $(TARGET)
