//
//    contour2ucm
//
//    Created by Di Yang, Vicent Rabaud, and Gary Bradski on 22/07/13.
//    Copyright (c) 2013 The Australian National University.
//    and Willow Garage inc.
//    All rights reserved.
//
//

#include "contour2ucm.h"
#define MAX_SIZE 100

using namespace std;

static contour_vertex* create_contour_vertex(int x, int y)
{
    contour_vertex* v = new contour_vertex[1];
    v->id = 0;
    v->is_subdivision = false;
    v->x = x;
    v->y = y;
    return v;
}

static contour_edge* create_contour_edge(contour_vertex & v_start,
        contour_vertex & v_end)
{
    contour_edge* e = new contour_edge[1];
    e->id = 0;
    e->contour_equiv_id = 0;
    e->is_completion = false;
    e->vertex_start = &v_start;
    e->vertex_end   = &v_end;
    e->vertex_start_enum = v_start.edges_start.size();
    e->vertex_end_enum = v_end.edges_end.size();
    v_start.edges_start.push_back(*e);
    v_end.edges_end.push_back(*e);
    return e;
}

cv::Point contour_vertex::point() const {
    return cv::Point(this->x, this->y);
}

int contour_edge::size() const {
    return this->x_coords.size();
}

cv::Point contour_edge::point(int n) const {
    return cv::Point(this->x_coords[n], this->y_coords[n]);
}

double contour_edge::length() const {
    cv::Point start = this->vertex_start->point();
    cv::Point end = this->vertex_end->point();
    double length = sqrt((end.x-start.x)*(end.x-start.x) + (end.y-start.y)*(end.y-start.y));
    return length;
}

namespace cv
{
void pb_normalize(const cv::Mat & input,
                  cv::Mat & output)
{
    float beta1 = -2.7487, beta2 = 11.1189, beta3 = 0.0602;
    input.copyTo(output);
    for(size_t i=0; i<output.rows; i++)
        for(size_t j=0; j<output.cols; j++) {
            float temp = output.at<float>(i,j);
            temp = 1/(1+exp(beta1+beta2*temp));
            temp = (temp-beta3)/(1-beta3);
            if(temp < 0)
                temp = 0;
            if(temp > 1)
                temp = 1;
            output.at<float>(i,j) = temp;
        }
}

void neighbor_exists_2D(const int* pos,
                        const int size_x, const int size_y,
                        cv::Mat & mask)
{
    //initial mask matrix
    mask = cv::Mat::ones(3, 3, CV_32SC1);
    mask.at<int>(1,1) = 0;

    //reset mask matrix according to current position ...
    if(pos[1]>=size_y-1) {
        int j = mask.cols-1;
        for(size_t i=0; i<mask.rows; i++)
            mask.at<int>(i, j) = 0;
    }
    if(pos[0]>=size_x-1) {
        int i = mask.rows-1;
        for(size_t j=0; j<mask.cols; j++)
            mask.at<int>(i, j) = 0;
    }

    if(pos[1]<= 0) {
        int j = 0;
        for(size_t i=0; i<mask.rows; i++)
            mask.at<int>(i, j) = 0;
    }
    if(pos[0] <= 0) {
        int i = 0;
        for(size_t j=0; j<mask.cols; j++)
            mask.at<int>(i, j) = 0;
    }
}

void neighbor_compare_2D(const int* pos,
                         const cv::Mat & labels,
                         const cv::Mat & mask,
                         cv::Mat & mask_cmp)
{
    mask_cmp = cv::Mat::zeros(3, 3, CV_32SC1);
    for(size_t i=0; i<mask.rows; i++)
        for(size_t j=0; j<mask.cols; j++) {
            int val = labels.at<int>(pos[0], pos[1]);
            if(mask.at<int>(i, j)==1)
                if(labels.at<int>(pos[0]+i-1, pos[1]+j-1)==val)
                    mask_cmp.at<int>(i,j)=1;
        }
}

bool is_vertex_2D(const cv::Mat & mask_cmp)
{
    int n, s, e, w, ne, nw, se, sw;
    n = mask_cmp.at<int>(1,2);
    s = mask_cmp.at<int>(1,0);
    e = mask_cmp.at<int>(2,1);
    w = mask_cmp.at<int>(0,1);

    ne = mask_cmp.at<int>(2,2);
    nw = mask_cmp.at<int>(0,2);
    se = mask_cmp.at<int>(2,0);
    sw = mask_cmp.at<int>(0,0);

    int n_adjacent = n + s + e + w;
    int n_diagonal = ne+ nw+ se+ sw;
    int nn = n_adjacent + n_diagonal;

    if(nn<=1)
        return true;
    else if(nn==2) {
        if((n && (ne || nw)) ||
                (s && (se || sw)) ||
                (e && (ne || se)) ||
                (w && (nw || sw)))
            return true;
        else
            return false;
    }
    else if((nn == 3) || (nn == 4)) {
        if((nw && n && w) || (n && ne && e) ||
                (w && sw && s) || (e && s && se))
            return true;
        else if
        ((nw && n && ne)|| (sw && s && se)||
                (nw && w && sw)|| (ne && e && se))
            return (nn==3);
        else if(n_adjacent >= 3 )
            return true;
        else if(n_diagonal >= 3 )
            return true;
        else if
        ((n && se && sw)|| (s && ne && nw)||
                (w && se && ne)|| (e && nw && sw))
            return true;
        else if
        ((nw && s && e) || (ne && s && w) ||
                (sw && n && e) || (se && n && w))
            return true;
        else
            return false;
    }
    else
        return(!((n_adjacent == 2) && ((n && s) || (e && w))));

}

void super_contour(const cv::Mat & input,
                   cv::Mat & output)
{
    cv::Mat H = cv::Mat::zeros(input.rows, input.cols-1, CV_32FC1);
    cv::Mat V = cv::Mat::zeros(input.rows-1, input.cols, CV_32FC1);
    output = cv::Mat::zeros(input.rows*2, input.cols*2, CV_32FC1);
    for(size_t i=0; i<input.rows; i++)
        for(size_t j=0; j<input.cols; j++) {
            if(j<input.cols-1)
                H.at<float>(i,j)=min(input.at<float>(i,j), input.at<float>(i, j+1));
            if(i<input.rows-1)
                V.at<float>(i,j)=min(input.at<float>(i,j), input.at<float>(i+1, j));
        }
    for(size_t i=0; i<output.rows; i++)
        for(size_t j=0; j<output.cols; j++) {
            if(i%2==0 && j%2==0)
                output.at<float>(i,j) = input.at<float>(i/2, j/2);
            if(i%2==0 && j%2==1 && (j-1)/2 < H.cols)
                output.at<float>(i,j) = H.at<float>(i/2, (j-1)/2);
            if(i%2==1 && j%2==0 && (i-1)/2 < V.rows)
                output.at<float>(i,j) = V.at<float>((i-1)/2, j/2);
            if(j == output.cols)
                output.at<float>(i,j) = max(output.at<float>(i,j), output.at<float>(i,j-1));
            if(i == output.rows)
                output.at<float>(i,j) = output.at<float>(i-1,j);
        }
}

void clean_watersheds(const cv::Mat & input,
                      cv::Mat & output,
                      cv::Mat & labels)
{
    input.copyTo(output);
    cv::Mat c = cv::Mat::zeros(input.rows, input.cols, CV_32SC1);
    for(size_t i=0; i<output.rows; i++)
        for(size_t j=0; j<output.cols; j++) {
            if(output.at<float>(i,j)==0.0)
                c.at<int>(i,j) = 1;
            /*if(i==0 || i==output.rows-1)
              c.at<int>(i,j) = 0;
            if(j==0 || j==output.cols-1)
            c.at<int>(i,j) = 0;*/
        }

    // Morphological clean up isolated pixel.
    cv::Mat mask;
    for(int i=0; i<c.rows; i++)
        for(int j=0; j<c.cols; j++) {
            bool is_break = false;
            int* pos = new int[2];
            pos[0] = i;
            pos[1] = j;
            mask = cv::Mat::zeros(3,3,CV_32SC1);
            neighbor_exists_2D(pos, c.rows, c.cols, mask);
            for(int x=0; x<3; x++) {
                for(int y=0; y<3; y++) {
                    if(mask.at<int>(x, y) == 1) {
                        if(c.at<int>(i+x-1, j+y-1) == c.at<int>(i,j)) {
                            is_break = true;
                            break;
                        }
                    }
                }
                if(is_break)
                    break;
            }
            if(!is_break)
                c.at<int>(i,j) = c.at<int>(i-1,j+1);
        }

    for(size_t i=0; i<c.rows; i++)
        for(size_t j=0; j<c.cols; j++) {
            if((c.at<int>(i,j)==0) && (output.at<float>(i,j)==0.0)) {
                float* vec = new float[4];
                int ind=0;
                vec[0] = max(output.at<float>(i-2,j-1), output.at<float>(i-1,j-2));
                vec[1] = max(output.at<float>(i+2,j-1), output.at<float>(i+1,j-2));
                vec[2] = max(output.at<float>(i+2,j+1), output.at<float>(i+1,j+2));
                vec[3] = max(output.at<float>(i-2,j+1), output.at<float>(i-1,j+2));

                for(size_t n=1; n<4; n++)
                    if(vec[ind]>vec[n])
                        ind = n;
                switch(ind) {
                case 0:
                    if(output.at<float>(i-2,j-1)<output.at<float>(i-1,j-2)) {
                        output.at<float>(i, j-1) = 0;
                        output.at<float>(i-1, j) = vec[0];
                    } else {
                        output.at<float>(i, j-1) = vec[0];
                        output.at<float>(i-1, j) = 0;
                    }
                    output.at<float>(i-1, j-1) = vec[0];
                    break;
                case 1:
                    if(output.at<float>(i+2, j-1) < output.at<float>(i+1, j-2)) {
                        output.at<float>(i, j-1) = 0;
                        output.at<float>(i+1, j) = vec[1];
                    } else {
                        output.at<float>(i, j-1) = vec[1];
                        output.at<float>(i+1, j) = 0;
                    }
                    output.at<float>(i+1, j-1) = vec[1];
                    break;
                case 2:
                    if(output.at<float>(i+2, j+1) < output.at<float>(i+1, j+2)) {
                        output.at<float>(i, j+1) = 0;
                        output.at<float>(i+1, j) = vec[2];
                    } else {
                        output.at<float>(i, j+1) = vec[2];
                        output.at<float>(i+1, j) = 0;
                    }
                    output.at<float>(i+1, j+1) = vec[2];
                    break;
                case 3:
                    if(output.at<float>(i-2, j+1) < output.at<float>(i-1, j+2)) {
                        output.at<float>(i, j+1) = 0;
                        output.at<float>(i-1, j) = vec[3];
                    } else {
                        output.at<float>(i, j+1) = vec[3];
                        output.at<float>(i-1, j) = 0;
                    }
                    output.at<float>(i-1, j+1) = vec[3];
                    break;
                }
            }
        }
    c = cv::Mat::zeros(input.rows, input.cols, CV_32SC1);
    for(size_t i=0; i<c.rows; i++)
        for(size_t j=0; j<c.cols; j++) {
            if(output.at<float>(i, j) == 0)
                c.at<int>(i, j) = -1;
        }

    int index = 1;
    for(size_t j=0; j<c.cols; j++)
        for(size_t i=0; i<c.rows; i++) {
            if(c.at<int>(i,j) == -1) {
                cv::Point seed;
                seed.x = j;
                seed.y = i;
                cv::floodFill(c, seed, cv::Scalar(index), 0, cv::Scalar(8));
                index++;
            }
        }
    labels = cv::Mat::zeros(c.rows/2, c.cols/2, CV_32SC1);
    vector<int> i_x;
    vector<int> i_y;
    for(size_t i=0; i<c.rows; i++)
        for(size_t j=0; j<c.cols; j++) {
            if((i%2 == 1) && (j%2==1)) {
                labels.at<int>((i-1)/2, (j-1)/2) = c.at<int>(i,j)-1;
                if( ((i-1)/2) < (c.rows/2-1) ||
                        ((j-1)/2) < (c.cols/2-1) ) {
                    if(labels.at<int>((i-1)/2, (j-1)/2) == -1) {
                        i_x.push_back((i-1)/2);
                        i_y.push_back((j-1)/2);
                    }
                }
            }
        }
    for(size_t i = 0; i<labels.rows; i++)
        labels.at<int>(i, labels.cols-1) = labels.at<int>(i, labels.cols-2);
    for(size_t j = 0; j<labels.cols; j++)
        labels.at<int>(labels.rows-1, j-1) = labels.at<int>(labels.rows-1, j-2);
    labels.at<int>(labels.rows-1, labels.cols-1) = labels.at<int>(labels.rows-2, labels.cols-2);

    if(i_x.size()) {
        for(size_t i=0; i<i_x.size(); i++) {
            int* pos = new int[2];
            int max_labels = -1;
            pos[0] = i_x[i];
            pos[1] = i_y[i];
            neighbor_exists_2D(pos, labels.rows, labels.cols, mask);
            for(size_t x=0; x<3; x++)
                for(size_t y=0; y<3; y++) {
                    if(mask.at<int>(x,y) == 1) {
                        if(max_labels < labels.at<int>(pos[0]+x-1, pos[1]+y-1))
                            max_labels = labels.at<int>(pos[0]+x-1, pos[1]+y-1);
                    }
                }
            labels.at<int>(pos[0], pos[1]) = max_labels;
        }
    }
}

void rot90(const cv::Mat & input,
           cv::Mat & output,
           int flag)
{
    cv::Mat temp;
    input.copyTo(output);
    if(flag == 1)
        output.t();
    output.copyTo(temp);
    for(size_t i=0; i<output.rows; i++)
        for(size_t j=0; j<output.cols; j++)
            output.at<float>(i, j) = temp.at<float>(output.rows-i-1, j);
    if(flag == -1)
        output.t();
}

void to_8(const cv::Mat & input,
          cv::Mat & output)
{
    input.copyTo(output);
    for(size_t i=0; i<output.rows-1; i++)
        for(size_t j=0; j<output.cols-1; j++)
            if(output.at<float>(i, j) > 0 &&
                    output.at<float>(i+1, j+1)>0 &&
                    output.at<float>(i, j+1)== 0 &&
                    output.at<float>(i+1, j)== 0)
                output.at<float>(i, j+1) = (output.at<float>(i, j) + output.at<float>(i+1, j+1))/2.0;
}


int connected_component(const cv::Mat & ws_bw,
                        cv::Mat & labels)
{
    cv::Mat mask;
    labels = cv::Mat::zeros(ws_bw.rows, ws_bw.cols, CV_32SC1);
    int next_label = 1;

    stack<int*> q;
    int* pos = new int[2];
    int* curr_pos = new int[2];

    for(size_t i=0; i<ws_bw.rows; i++)
        for(size_t j=0; j<ws_bw.cols; j++) {
            int val = ws_bw.at<int>(i,j);
            if((val != 0) &&
                    (labels.at<int>(i,j) == 0)) {
                labels.at<int>(i,j)=next_label;

                //record current position
                pos[0] = i;
                pos[1] = j;
                q.push(pos);
                while(!q.empty()) {
                    curr_pos = q.top(); //load current position
                    q.pop();            //clear current position from stack
                    neighbor_exists_2D(curr_pos, ws_bw.rows, ws_bw.cols, mask);
                    //check neighborhood
                    for(size_t m_i=0; m_i<mask.rows; m_i++)
                        for(size_t m_j=0; m_j<mask.cols; m_j++) {
                            if(mask.at<int>(m_i, m_j) == 1) {
                                int ind_x = curr_pos[0]+m_i-1;
                                int ind_y = curr_pos[1]+m_j-1;
                                int* neigh_pos = new int[2];
                                if((ws_bw.at<int>(ind_x, ind_y) == val) &&
                                        (labels.at<int>(ind_x, ind_y) == 0)) {
                                    labels.at<int>(ind_x, ind_y) = next_label;
                                    neigh_pos[0] = ind_x;
                                    neigh_pos[1] = ind_y;
                                    q.push(neigh_pos);
                                }
                            }
                        }
                }//end_while
                next_label++;
            }
        }
    next_label--;
    return next_label;
}

void contour_side(const cv::Mat & ws_bw,
                  cv::Mat & labels,
                  cv::Mat & _is_edge,
                  cv::Mat & _is_vertex,
                  cv::Mat & _assignment,
                  vector<contour_vertex> & _vertices,
                  vector<contour_edge> & _edges)
{
    int num_labels = connected_component(ws_bw, labels);
    int rows = ws_bw.rows, cols = ws_bw.cols;
    int* label_x = new int[num_labels];
    int* label_y = new int[num_labels];
    bool* has_vertex = new bool[num_labels];
    int* pos = new int[2];

    cv::Mat mask, mask_cmp;
    vector< vector<contour_edge> > _edges_equiv;

    _assignment = cv::Mat::zeros(rows, cols, CV_32SC1);
    _is_vertex = cv::Mat::zeros(rows, cols, CV_32SC1);
    _is_edge   = cv::Mat::zeros(rows, cols, CV_32SC1);

    //--------------------- Vertex Assignment ----------------

    for(size_t i=0; i<rows; i++)
        for(size_t j=0; j<cols; j++) {
            int label = labels.at<int>(i,j);
            if(label != 0) {
                pos[0] = i;
                pos[1] = j;
                neighbor_exists_2D(pos, rows, cols, mask);
                neighbor_compare_2D(pos, labels, mask, mask_cmp);
                if(is_vertex_2D(mask_cmp)) {
                    contour_vertex* v = create_contour_vertex(i,j);
                    v->id = _vertices.size();
                    _assignment.at<int>(i,j) = v->id;
                    _vertices.push_back(*v);
                    delete[] v;
                    _is_vertex.at<int>(i, j) = 1;
                    _is_edge.at<int>(i, j) = 0;
                    has_vertex[label-1] = true;
                }
                label_x[label-1]=i;
                label_y[label-1]=j;
            }
        }

    for(size_t i=0; i<num_labels; i++) {
        if(!has_vertex[i]) {
            contour_vertex* v = create_contour_vertex(label_x[i], label_y[i]);
            v->id = _vertices.size();
            _assignment.at<int>(label_x[i], label_y[i])=v->id;
            _vertices.push_back(*v);
            delete[] v;
            _is_vertex.at<int>(label_x[i], label_y[i]) = 1;
            _is_edge.at<int>(label_x[i], label_y[i]) = 0;
            has_vertex[i] = true;
        }
    }

    //------------- EDGE ASSIGNMENT ---------------------

    for(size_t v_id = 0; v_id < _vertices.size(); v_id++) {
        contour_vertex* v = &_vertices[v_id];
        int label = labels.at<int>(v->x, v->y);
        int q_x[8] = { 0 };
        int q_y[8] = { 0 };
        int n_neighbors = 0;
        int* pos = new int[2];
        pos[0] = v->x;
        pos[1] = v->y;
        neighbor_exists_2D(pos, rows, cols, mask);
        neighbor_compare_2D(pos, labels, mask, mask_cmp);
        for(size_t i = 0; i<mask_cmp.rows; i++)
            for(size_t j = 0; j<mask_cmp.cols; j++) {
                if(mask_cmp.at<int>(i, j) == 1) {
                    q_x[n_neighbors] = v->x + i - 1;
                    q_y[n_neighbors] = v->y + j - 1;
                    n_neighbors++;
                }
            }

        for(size_t n = 0; n < n_neighbors; n++) {
            int nx = q_x[n], ny = q_y[n];
            if(_is_vertex.at<int>(nx, ny)) {
                if(_assignment.at<int>(nx, ny) > v_id) {
                    contour_vertex * v_end = &_vertices[_assignment.at<int>(nx, ny)];
                    contour_edge* e      = create_contour_edge(*v, *v_end);
                    e->id                = _edges.size();
                    e->contour_equiv_id  = e->id;
                    // create edge equivalence class
                    vector<contour_edge> e_equiv;
                    e_equiv.push_back(*e);
                    // add edge
                    _edges.push_back(*e);
                    _edges_equiv.push_back(e_equiv);
                    e_equiv.clear();
                    delete[] e;
                }
            }
            else if(!_is_edge.at<int>(nx, ny)) {
                // temporarily mark start vertex as inaccessible
                _is_edge.at<int>(v->x, v->y) = 1;
                int e_id = _edges.size();
                bool endpoint_is_new_vertex = false;
                vector<int> e_x;
                vector<int> e_y;
                do {
                    e_x.push_back(nx);
                    e_y.push_back(ny);
                    _is_edge.at<int>(nx, ny) = 1;
                    _assignment.at<int>(nx, ny) = e_id;
                    pos[0] = nx;
                    pos[1] = ny;
                    neighbor_exists_2D(pos, rows, cols, mask);
                    mask_cmp = cv::Mat::zeros(3, 3, CV_32SC1);
                    for(size_t i=0; i<3; i++)
                        for(size_t j=0; j<3; j++) {
                            if(mask.at<int>(i,j) == 1) {
                                int i_x = nx+i-1, i_y = ny+j-1;
                                if((labels.at<int>(i_x, i_y)==label) &&
                                        (!_is_edge.at<int>(i_x, i_y)))
                                    mask_cmp.at<int>(i, j) = 1;
                            }
                        }

                    // check neighborhood
                    if(mask_cmp.at<int>(0,0)) {
                        nx--;
                        ny--;
                        if(!_is_vertex.at<int>(nx, ny))
                            continue;
                        else
                            break;
                    }

                    if(mask_cmp.at<int>(0,1)) {
                        nx--;
                        ny;
                        if(!_is_vertex.at<int>(nx, ny))
                            continue;
                        else
                            break;
                    }

                    if(mask_cmp.at<int>(0,2)) {
                        nx--;
                        ny++;
                        if(!_is_vertex.at<int>(nx, ny))
                            continue;
                        else
                            break;
                    }

                    if(mask_cmp.at<int>(1,0)) {
                        nx;
                        ny--;
                        if(!_is_vertex.at<int>(nx, ny))
                            continue;
                        else
                            break;
                    }

                    if(mask_cmp.at<int>(1,2)) {
                        nx;
                        ny++;
                        if(!_is_vertex.at<int>(nx, ny))
                            continue;
                        else
                            break;
                    }

                    if(mask_cmp.at<int>(2,0)) {
                        nx++;
                        ny--;
                        if(!_is_vertex.at<int>(nx, ny))
                            continue;
                        else
                            break;
                    }

                    if(mask_cmp.at<int>(2,1)) {
                        nx++;
                        ny;
                        if(!_is_vertex.at<int>(nx, ny))
                            continue;
                        else
                            break;
                    }

                    if(mask_cmp.at<int>(2,2)) {
                        nx++;
                        ny++;
                        if(!_is_vertex.at<int>(nx, ny))
                            continue;
                        else
                            break;
                    }
                    endpoint_is_new_vertex = true;
                    _is_edge.at<int>(nx, ny) = 0;
                } while(!endpoint_is_new_vertex);
                // remove inaccesssible mark from start vertex
                _is_edge.at<int>(v->x, v->y) = 0;

                // add endpoint as vertex (if needed) probably never get here
                if(endpoint_is_new_vertex) {
                    contour_vertex * v_end = create_contour_vertex(nx, ny);
                    v_end->id = _vertices.size();
                    _assignment.at<int>(nx, ny) = v_end->id;
                    _vertices.push_back(*v_end);
                    delete[] v_end;
                    _is_vertex.at<int>(nx, ny) = 1;
                }

                // create edge and set its identity
                contour_vertex * v_e = &_vertices[_assignment.at<int>(nx, ny)];
                contour_edge* e = create_contour_edge(*v, *v_e);
                e->id = e_id;
                e->contour_equiv_id = e_id;
                int n_edge_points = (endpoint_is_new_vertex ? (e_x.size()-1) : e_x.size());
                e->x_coords.resize(n_edge_points);
                e->y_coords.resize(n_edge_points);
                for(size_t i = 0; i<n_edge_points; i++) {
                    e->x_coords[i] = e_x[i];
                    e->y_coords[i] = e_y[i];
                }
                vector<contour_edge> e_equiv;
                e_equiv.push_back(*e);
                _edges.push_back(*e);
                delete[] e;
                _edges_equiv.push_back(e_equiv);
                e_equiv.clear();
            }
        }
    }


    int n_vertices = _vertices.size();
    for(size_t v_id=0, n=0; n<n_vertices; n++) {
        contour_vertex v;
        v = _vertices.back();
        _vertices.pop_back();
        if((v.edges_start.empty()) && (v.edges_end.empty()))
            _is_vertex.at<int>(v.x, v.y) = 0;
        else {
            v.id = v_id++;
            _assignment.at<int>(v.x, v.y) = v.id;
            _vertices.push_back(v);
        }
    }
}

void fit_contour(const cv::Mat & ws_bw,
                 cv::Mat & labels,
                 cv::Mat & _is_edge,
                 cv::Mat & _is_vertex,
                 cv::Mat & edges_endpoints,
                 vector<contour_vertex> & _vertices,
                 vector<contour_edge> & _edges)
{
    cv::Mat _assignment;
    contour_side(ws_bw, labels, _is_edge, _is_vertex, _assignment, _vertices, _edges);

    int n_edges = _edges.size();
    edges_endpoints = cv::Mat::zeros(n_edges, 2, CV_32SC1);

    for(size_t i = 0; i<n_edges; i++) {
        edges_endpoints.at<int>(i,1) = _edges[i].vertex_start->id;
        edges_endpoints.at<int>(i,2) = _edges[i].vertex_end->id;
    }
}

void creat_finest_partition(const cv::Mat & gPb,
                            const vector<cv::Mat> & gPb_ori,
                            cv::Mat & ws_wt)
{
    cv::Mat edges_endpoints, _is_vertex, _is_edge, labels;
    vector<contour_vertex> _vertices;
    vector<contour_edge> _edges;
    double minVal, maxVal;

    cv::Mat temp = cv::Mat::zeros(gPb.rows, gPb.cols, CV_32FC1);
    cv::Mat ws_bw = cv::Mat::ones(gPb.rows, gPb.cols, CV_32FC1);

    cv::minMaxIdx(gPb, &minVal, &maxVal, NULL, NULL);
    cv::addWeighted(gPb, 1.0, ws_bw, -minVal, 0.0, temp);
    cv::multiply(temp, ws_bw, temp, 255.0/(maxVal-minVal));
    temp.convertTo(temp, CV_64FC1);
    cv::watershedFull(temp, 1, ws_wt);

    for(size_t i=0; i<ws_wt.rows; i++)
        for(size_t j=0; j<ws_wt.cols; j++)
            if(ws_wt.at<int>(i,j) > 0)
                ws_bw.at<int>(i,j)=0;

    fit_contour(ws_bw, labels, _is_edge, _is_vertex, edges_endpoints,
                _vertices, _edges);
    ws_wt = cv::Mat::zeros(gPb.rows, gPb.cols, CV_32FC1);

    int n_edges = _edges.size();
    for(size_t i=0; i<n_edges; i++) {
        cv::Point v1, v2;
        double ang;
        int orient;
        v1.x = _vertices[edges_endpoints.at<int>(i,1)].x;
        v1.y = _vertices[edges_endpoints.at<int>(i,1)].y;

        v2.x = _vertices[edges_endpoints.at<int>(i,2)].x;
        v2.y = _vertices[edges_endpoints.at<int>(i,2)].y;

        if(v1.y == v2.y)
            ang = 0.5*M_PI;
        else
            ang = atan((v1.x-v2.x) / (v1.y-v2.y));

        if( (ang<-78.75) || (ang>=78.75) )
            orient = 1;
        else if( (ang<78.75) && (ang>=56.25) )
            orient = 2;
        else if( (ang<56.25) && (ang>=33.75) )
            orient = 3;
        else if( (ang<33.75) && (ang>=11.25) )
            orient = 4;
        else if( (ang<11.25) && (ang>=-11.25))
            orient = 5;
        else if( (ang<-11.25) && (ang>=-33.75))
            orient = 6;
        else if( (ang<-33.75) && (ang>=-56.25))
            orient = 7;
        else if( (ang<-56.25) && (ang>=-78.75))
            orient = 8;

        for(size_t j=0; j<_edges[i].x_coords.size(); j++)
            ws_wt.at<float>(_edges[i].x_coords[j], _edges[i].y_coords[j]) =
                max(gPb_ori[orient].at<float>(_edges[i].x_coords[j], _edges[i].y_coords[j]), ws_wt.at<float>(_edges[i].x_coords[j], _edges[i].y_coords[j]));

        ws_wt.at<float>(v1.x, v1.y) = max(gPb_ori[orient].at<float>(v1.x, v1.y), ws_wt.at<float>(v1.x, v1.y));
        ws_wt.at<float>(v2.x, v2.y) = max(gPb_ori[orient].at<float>(v2.x, v2.y), ws_wt.at<float>(v2.x, v2.y));
    }

    //clean up
    _vertices.clear();
    _edges.clear();
}

void contour2ucm(const cv::Mat & gPb,
                 const vector<cv::Mat> & gPb_ori,
                 cv::Mat & ucm,
                 bool label)
{
    bool flag = label ? DOUBLE_SIZE : SINGLE_SIZE;
    cv::Mat ws_wt8, ws_wt2, labels, ws_wt;
    creat_finest_partition(gPb, gPb_ori, ws_wt);

    rot90(ws_wt, ws_wt8, 1);
    to_8(ws_wt8, ws_wt8);
    rot90(ws_wt8, ws_wt8, -1);
    to_8(ws_wt8, ws_wt8);
    super_contour(ws_wt8, ws_wt2);
    clean_watersheds(ws_wt2, ws_wt2, labels);

    cv::copyMakeBorder(ws_wt2, ws_wt2, 0, 1, 0, 1, cv::BORDER_REFLECT);
    cv::ucm_mean_pb(ws_wt2, labels, ucm, flag);
    pb_normalize(ucm, ucm);
}
}
