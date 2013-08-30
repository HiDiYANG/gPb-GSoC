CC = g++
CFLAGS = `pkg-config --cflags opencv` -I./include/gPb -I./include/sPb -I./include/seg

LIBS = `pkg-config --libs opencv` -L/opt/local/lib -larpack -lparpack -lgfortran

SRC = 	src/main.cpp 		   \
	src/gPb/globalPb.cpp       \
	src/gPb/Filters.cpp        \
	src/sPb/buildW.cpp         \
	src/sPb/ic.cpp             \
	src/sPb/affinity.cpp       \
	src/sPb/smatrix.cpp        \
	src/sPb/normCut.cpp        \
	src/seg/watershed.cpp      \
	src/seg/VisWatershed.cpp   \
	src/seg/contour2ucm.cpp    \
	src/seg/ucm_mean_pb.cpp    \
	src/seg/uvt.cpp 

OBJ = gPb

program:
	$(CC) -g -o $(OBJ) $(SRC) $(CFLAGS) $(LIBS)

clean:
	rm $(OBJ)