gslinc = $(shell gsl-config --cflags)
gsllib = $(shell gsl-config --libs)

libs: libsfac.so

libsfac.so : r-matrix-simple.c
	gcc r-matrix-simple.c -Ofast -fPIC -shared $(gslinc) $(gsllib) -o libsfac.so
