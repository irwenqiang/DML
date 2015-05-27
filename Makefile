lr:lr.o
	g++ -g -o lr lr.o
lr.o:lr.cpp lr.h
	g++ -g -c lr.cpp
clean:
	rm -f *~ lr lr.o
