reduction: reduction.o
	g++ -o reduction reduction.o -framework opencl
	rm *.o
reduction.o: reduction.cpp
	g++ -c reduction.cpp

clean:
	rm reduction