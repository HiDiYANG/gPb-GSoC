CC = g++
CFLAGS = `pkg-config --cflags opencv` -I./headers -I./buildW
LIBS = `pkg-config --libs opencv`

SRC = src/example.cc src/globalPb.cc src/cv_lib_filters.cc buildW/buildW.cc buildW/ic.cc buildW/affinity.cc buildW/smatrix.cc
OBJ = test
program:
	$(CC) -o $(OBJ) $(SRC) $(CFLAGS) $(LIBS)
