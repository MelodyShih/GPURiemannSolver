main: main.o
	g++ -o main main.o -framework opencl
	rm *.o
main.o: main.cpp
	g++ -c main.cpp

main2: main2.o
	g++ -o main2 main2.o -framework opencl
	rm *.o
main2.o: main2.cpp
	g++ -c main2.cpp

clean:
	rm main
	rm main2