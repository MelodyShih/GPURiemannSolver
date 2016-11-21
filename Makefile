acoustic: acoustic.o
	g++ -o $@ $< -framework opencl
	rm *.o
acoustic.o: acoustic.cpp
	g++ -c $<

acoustic_v2: acoustic_v2.o
	g++ -o $@ $< -framework opencl
	rm *.o
acoustic_v2.o: acoustic_v2.cpp
	g++ -c $<

euler_v2: euler_v2.o
	g++ -o $@ $< -framework opencl
	rm *.o
euler_v2.o: euler_v2.cpp
	g++ -c $<

clean:
	rm acoustic
	rm acoustic_v2
	rm euler_v2