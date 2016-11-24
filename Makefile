acoustic: acoustic.o
	g++ -o $@ $< -L/usr/local/cuda-7.5/lib64 -lOpenCL
	rm *.o
acoustic.o: acoustic.cpp
	g++  -c $< -I/usr/local/cuda-7.5/include

acoustic_v2: acoustic_v2.o
	g++ -o $@ $< -L/usr/local/cuda-7.5/lib64 -lOpenCL
	rm *.o
acoustic_v2.o: acoustic_v2.cpp
	g++ -c $< -I/usr/local/cuda-7.5/include

euler_v2: euler_v2.o
	g++ -o $@ $< -L/usr/local/cuda-7.5/lib64 -lOpenCL
	rm *.o 
euler_v2.o: euler_v2.cpp
	g++ -c $< -I/usr/local/cuda-7.5/include

clean:
	rm acoustic
	rm acoustic_v2
	rm euler_v2