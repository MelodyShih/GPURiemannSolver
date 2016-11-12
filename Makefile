main: main.o
	g++ -o main main.o -framework opencl
	rm *.o
main.o: main.cpp
	g++ -c main.cpp

clean:
	rm main