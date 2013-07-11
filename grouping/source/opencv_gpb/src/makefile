CC = g++
CFLAGS = `pkg-config --cflags opencv`
LIBS = `pkg-config --libs opencv`

SRC = example.cc globalPb.cc cv_lib_filters.cc buildW/buildW.cc buildW/ic.cc buildW/affinity.cc buildW/smatrix.cc
OBJ = test
program:
	$(CC) -o $(OBJ) $(SRC) $(CFLAGS) $(LIBS)
