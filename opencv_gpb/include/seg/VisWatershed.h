//
//  watertest.h
//  ImpWshed
//
//  Created by Di Yang on 1/03/12.
//  Copyright (c) 2012 The Australian National University. All rights reserved.
//

#include <iostream>
#include <opencv/cv.h>
#include <opencv/highgui.h>

typedef struct CvWSNode
{
	struct CvWSNode* next;
	int mask_ofs;
	int img_ofs;
}
CvWSNode;

typedef struct CvWSQueue
{
	CvWSNode* first;
	CvWSNode* last;
}
CvWSQueue;

static CvWSNode* icvAllocWSNodes( CvMemStorage* storage );
 
void VisWatershed( const CvArr* srcarr,
                         CvArr* dstarr,
                   int sz,
                   double C );
namespace cv{
  void viswatershed( InputArray _src, 
		     InputOutputArray markers, 
		     int sz, 
		     double C);
}
