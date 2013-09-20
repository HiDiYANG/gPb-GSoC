//
//  watertest.cpp
//  ImpWshed
//
//  Created by Di Yang on 1/03/12.
//  Copyright (c) 2012 The Australian National University. All rights reserved.
//

#include "VisWatershed.h"

/****************************************************************************
 *                           Improved Watershed                             *    
 \****************************************************************************/

static CvWSNode*
icvAllocWSNodes( CvMemStorage* storage )
{
	CvWSNode* n = 0;	
	int i, count = (storage->block_size - sizeof(CvMemBlock))/sizeof(*n) - 1;
	n = (CvWSNode*)cvMemStorageAlloc( storage, count*sizeof(*n) );
	for( i = 0; i < count-1; i++ )
		n[i].next = n + i + 1;
	n[count-1].next = 0;
	return n;
}

void VisWatershed( const CvArr* srcarr,
                         CvArr* dstarr,
                   int sz,
                   double C)
{
	const int IN_QUEUE = -2;
	const int WSHED = -1;
	const int NQ = 256;
	//const double C = 0.02 ;//0.002;
	//const int sz = 2;
	const int thres = 256;
	const int block = (2*sz+1)*(2*sz+1);
	CvMemStorage* storage = 0;
	
	CvMat sstub, *src;
	CvMat dstub, *dst;
	CvSize size;
	CvWSNode* free_node = 0, *node;
	CvWSQueue q[NQ];
	int active_queue;
	int i, j;
	int db, dg, dr;
	int* mask;
	uchar* img;
	int mstep, istep;
	int subs_tab[513];
	
	// MAX(a,b) = b + MAX(a-b,0)
#define ws_max(a,b) ((b) + subs_tab[(a)-(b)+NQ])
	// MIN(a,b) = a - MAX(a-b,0)
#define ws_min(a,b) ((a) - subs_tab[(a)-(b)+NQ])
	
#define ws_push(idx,mofs,iofs)  \
{                               \
if( !free_node )            \
free_node = icvAllocWSNodes( storage );\
node = free_node;           \
free_node = free_node->next;\
node->next = 0;             \
node->mask_ofs = mofs;      \
node->img_ofs = iofs;       \
if( q[idx].last )           \
q[idx].last->next=node; \
else                        \
q[idx].first = node;    \
q[idx].last = node;         \
}
    
#define ws_pop(idx,mofs,iofs)   \
{                               \
node = q[idx].first;        \
q[idx].first = node->next;  \
if( !node->next )           \
q[idx].last = 0;        \
node->next = free_node;     \
free_node = node;           \
mofs = node->mask_ofs;      \
iofs = node->img_ofs;       \
}
	
#define c_diff(ptr1,ptr2,diff)      \
{                                   \
db = abs((ptr1)[0] - (ptr2)[0]);\
dg = abs((ptr1)[1] - (ptr2)[1]);\
dr = abs((ptr1)[2] - (ptr2)[2]);\
diff = ws_max(db,dg);           \
diff = ws_max(diff,dr);         \
assert( 0 <= diff && diff <= 255 ); \
}
    
	src = cvGetMat( srcarr, &sstub );
	dst = cvGetMat( dstarr, &dstub );
	
	if( CV_MAT_TYPE(src->type) != CV_8UC3 ){
		printf("Only 8-bit, 3-channel input images are supported");
		exit;
	}
	//CV_ERROR( CV_StsUnsupportedFormat, "Only 8-bit, 3-channel input images are supported" );
	
	if( CV_MAT_TYPE(dst->type) != CV_32SC1 ){
		printf("Only 32-bit, 1-channel output images are supported");
		exit;
	}
	//CV_ERROR( CV_StsUnsupportedFormat, "Only 32-bit, 1-channel output images are supported" );
	
	if( !CV_ARE_SIZES_EQ( src, dst )){
		printf("The input and output images must have the same size");
		exit;
	}
	//CV_ERROR( CV_StsUnmatchedSizes, "The input and output images must have the same size" );
	
	size = cvGetMatSize(src);
	
	storage = cvCreateMemStorage();
	
	istep = src->step;
	img = src->data.ptr;
	mstep = dst->step / sizeof(mask[0]);
	mask = dst->data.i;
	
	memset( q, 0, NQ*sizeof(q[0]) );
	
	for( i = 0; i < 256; i++ )
		subs_tab[i] = 0;
	for( i = 256; i <= 512; i++ )
		subs_tab[i] = i - 256;
	
	// draw a pixel-wide border of dummy "watershed" (i.e. boundary) pixels
    for( i = 0; i < sz; i++ ) 
        for( j = 0; j < size.width; j++ ){
            mask[j+i*mstep] = mask[j + mstep*(size.height-i-1)] = WSHED;
        }
    
	mask += (sz-1)*mstep;
    img += (sz-1)*istep;
	// initial phase: put all the neighbor pixels of each marker to the ordered queue -
	// determine the initial boundaries of the basins
	for( i = sz; i < size.height-sz; i++ )
	{
		img += istep; mask += mstep;
        for( int n=0; n < sz; n++ )
		    mask[0+n] = mask[size.width-n-1] = WSHED;
        
		for( j = sz; j < size.width-sz; j++ )
		{
			int* m = mask + j;
			if( m[0] < 0 ) m[0] = 0;
			if( m[0] == 0 && (m[-1] > 0 || m[1] > 0 || m[-mstep] > 0 || m[mstep] > 0) )
			{
				uchar* ptr = img + j*3;
				int idx = 256, t;
				if( m[-1] > 0 )
					c_diff( ptr, ptr - 3, idx );
				if( m[1] > 0 )
				{
					c_diff( ptr, ptr + 3, t );
					idx = ws_min( idx, t );
				}
				if( m[-mstep] > 0 )
				{
					c_diff( ptr, ptr - istep, t );
					idx = ws_min( idx, t );
				}
				if( m[mstep] > 0 )
				{
					c_diff( ptr, ptr + istep, t );
					idx = ws_min( idx, t );
				}
				assert( 0 <= idx && idx <= 255 );
                ws_push( idx, i*mstep + j, i*istep + j*3 );
				m[0] = IN_QUEUE;
			}
		}
	}
    //cvSaveImage("../mask.png", dst);
	
	// find the first non-empty queue
	for( i = 0; i < NQ; i++ )
		if( q[i].first )
			break;
    
	// if there is no markers, exit immediately
	if( i == NQ )
		exit;
	
	active_queue = i;
	img = src->data.ptr;
	mask = dst->data.i;
	
	// recursively fill the basins
	for(;;)
	{
		int mofs, iofs;
		int lab = 0, t, t1, b;
		int* m;
		uchar* ptr;
		
		/*if (active_queue == 0) {
         break;
         }*/
		
		if( q[active_queue].first == 0 )
		{
			for( i = active_queue+1; i < NQ; i++ )
				if( q[i].first )
					break;
			
			if( i == NQ )
				break;
			active_queue = i;
		}
		
		ws_pop( active_queue, mofs, iofs );
		
		m = mask + mofs;
		ptr = img + iofs;
		t = m[-1];
		if( t > 0 ) lab = t;
		t = m[1];
		if( t > 0 )
			if( lab == 0 ) lab = t;
			else if( t != lab ) lab = WSHED;
		t = m[-mstep];
		if( t > 0 )
			if( lab == 0 ) lab = t;
			else if( t != lab ) lab = WSHED;
		t = m[mstep];
		if( t > 0 )
			if( lab == 0 ) lab = t;
			else if( t != lab ) lab = WSHED;
        
        assert( lab != 0 );
		m[0] = lab;
		if( lab == WSHED )
			continue;
		
		if( m[-1] == 0 )
		{
			c_diff( ptr, ptr - 3, t );
            if( t < thres ){
                double a = 0;
                for( int i=-sz; i<=sz; i++ )
                    for( int j=-sz; j<=sz; j++ ){
                        c_diff( ptr-3, ptr-i*3+j*istep, t1);
                        a = a + exp(-C*double(t1));
                    }
                b = int(log(double(block)-a+1.0)/C+0.5);   
                t=t+b;
            }
            if( t>255 ) t=255;
  			ws_push( t, mofs - 1, iofs - 3 );
			active_queue = ws_min( active_queue, t );
			m[-1] = IN_QUEUE;
		}
		if( m[1] == 0 )
		{
			c_diff( ptr, ptr + 3, t );
            if( t < thres ){
                double a = 0;
                for( int i=-sz; i<=sz; i++ )
                    for( int j=-sz; j<=sz; j++ ){
                        c_diff( ptr+3, ptr+i*3+j*istep, t1);
                        a = a + exp(-C*double(t1));
                    }
                b = int(log(double(block)-a+1.0)/C+0.5);
                t=t+b;
            }
            if( t>255 ) t=255;
			ws_push( t, mofs + 1, iofs + 3 );
			active_queue = ws_min( active_queue, t );
			m[1] = IN_QUEUE;
		}
		if( m[-mstep] == 0 )
		{
			c_diff( ptr, ptr-istep, t );
			if( t < thres ){
                double a = 0;
                for( int i=-sz; i<=sz; i++ )
                    for( int j=-sz; j<=sz; j++ ){
                        c_diff( ptr-istep, ptr+j*3-i*istep, t1 );
                        a = a + exp(-C*double(t1));
                    }
                b = int(log(double(block)-a+1.0)/C+0.5);
                t=t+b;
            }
            if( t>255 ) t=255;
			ws_push( t, mofs - mstep, iofs - istep );
			active_queue = ws_min( active_queue, t );
			m[-mstep] = IN_QUEUE;
		}
		if( m[mstep] == 0 )
		{
			c_diff( ptr, ptr+istep, t );
            if( t < thres ){
                double a = 0;
                for( int i=-sz; i<=sz; i++ )
                    for( int j=-sz; j<=sz; j++ ){
                        c_diff( ptr+istep, ptr+j*3+i*istep, t1 );
                        a = a + exp(-C*double(t1));
                    }
                b = int(log(double(block)-a+1.0)/C+0.5);
                t=t+b;
            }
            if( t>255 ) t=255;
			ws_push( t, mofs + mstep, iofs + istep );
			active_queue = ws_min( active_queue, t );
			m[mstep] = IN_QUEUE;
		}
    }
    
    cvReleaseMemStorage( &storage );
}

namespace cv{
  void viswatershed( InputArray _src, InputOutputArray markers, int sz, double c)
  {
    Mat src = _src.getMat();
    CvMat c_src = _src.getMat(), c_markers = markers.getMat();
    VisWatershed( &c_src, &c_markers, sz, c);
  }
}
