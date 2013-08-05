CC = g++
CFLAGS = `pkg-config --cflags opencv` -I./headers -I./buildW

LIBS = `pkg-config --libs opencv`

SRC = 	src/main.cc src/globalPb.cc src/cv_lib_filters.cc src/watershed.cpp src/contour2ucm.cc src/ucm_mean_pb.cpp src/uvt.cpp buildW/buildW.cc buildW/ic.cc buildW/affinity.cc buildW/smatrix.cc
OBJ = test
program:
	$(CC) -o $(OBJ) $(SRC) $(CFLAGS) $(LIBS)

clean:
	rm $(OBJ)
	rm *.txt
