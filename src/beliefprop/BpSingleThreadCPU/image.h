/*
Copyright (C) 2006 Pedro Felzenszwalb

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307 USA
*/

/* a simple image class */

#ifndef IMAGE_H
#define IMAGE_H

#include <cstring>
#include <memory>

/**
 * @brief Class and structs in single-thread CPU bp implementation by Pedro
 * Felzenwalb available at https://cs.brown.edu/people/pfelzens/bp/index.html
 */
namespace bp_single_thread_imp {

template <class T>
class image {
 public:
  /* create an image */
  image(int width, int height, bool init = true);

  /* delete an image */
  ~image();

  /* init an image */
  void init(const T &val);

  /* copy an image */
  image<T> *copy() const;
  
  /* get the width of an image. */
  int width() const { return w; }
  
  /* get the height of an image. */
  int height() const { return h; }
  
  /* image data. */
  T *data;
  
  /* row pointers. */
  T **access;
  
 private:
  int w, h;
};

template <class T>
class imageWDisp {
 public:
  /* create an image */
  imageWDisp(int width, int height, int num_disp_vals, bool init = true);

  /* delete an image */
  ~imageWDisp();

  /* init an image */
  void init(const T &val);

  /* copy an image */
  imageWDisp<T> *copy() const;
  
  /* get the width of an image. */
  int width() const { return w; }
  
  /* get the height of an image. */
  int height() const { return h; }

  /* get the number of disparity values of an image. */
  int DisparityVals() const { return disp_vals; }
  
  /* image data. */
  T *data;
  
  /* row pointers. */
  T **access;
  
 private:
  int w, h, disp_vals;
};

//uncomment for indexing where x-value is last when getting index (as opposed
//to disparity value); indexing this way (along with splitting up messages/data
//costs into separate vectors by "checkerboard") needed for SIMD processing but
//seems to be slower when not using SIMD
//#define INDEXING_W_X_VAL_LAST

template <class T>
class BpVector {
public:
  //default constructor
  BpVector() {}

  /* create BpVector with specified width, height, and number of disparity values */
  BpVector(unsigned int width, unsigned int height, unsigned int num_disp_vals) :
    width_{width}, height_{height}, num_disp_vals_(num_disp_vals)
  {
    //bp_vals_ = std::vector<T>(width_*height_*num_disp_vals_);
    bp_vals_ = std::make_unique<T[]>(width_*height_*num_disp_vals_);
  }

  inline T& operator()(unsigned int x, unsigned int y, unsigned int disparity) noexcept
  {
#if defined(INDEXING_W_X_VAL_LAST)
    return bp_vals_[(y * (width_ * num_disp_vals_)) + (width_ * disparity) + x];
#else
    return bp_vals_[(y * width_ * num_disp_vals_) + (x * num_disp_vals_) + disparity];
#endif //INDEXING_W_X_VAL_LAST
  }

  inline const T& operator()(unsigned int x, unsigned int y, unsigned int disparity) const noexcept
  {
#if defined(INDEXING_W_X_VAL_LAST)
    return bp_vals_[(y * (width_ * num_disp_vals_)) + (width_ * disparity) + x];
#else
    return bp_vals_[(y * width_ * num_disp_vals_) + (x * num_disp_vals_) + disparity];
#endif //INDEXING_W_X_VAL_LAST
  }

  inline T* DispValsPtr(unsigned int x, unsigned int y) const noexcept
  {
#if defined(INDEXING_W_X_VAL_LAST)
    return &(bp_vals_[(y * (width_ * num_disp_vals_)) + (width_ * 0) + x]);
#else
    return &(bp_vals_[(y * width_ * num_disp_vals_) + (x * num_disp_vals_)]);
#endif //INDEXING_W_X_VAL_LAST
  }

  /**
   * @brief Get a vector with the values corresponding to each disparity
   * 
   * @param x 
   * @param y 
   * @return std::vector<T> 
   */
  std::vector<T> ValsEachDisparity(unsigned int x, unsigned int y) const {
    std::vector<T> vals_each_disp(num_disp_vals_);
    //auto vals_each_disp = std::make_unique<T[]>(num_disp_vals_);
    //go through each disparity and add values corresponding to each disparity
    //to vector
    for (unsigned int disp = 0; disp < num_disp_vals_; disp++) {
      //vals_each_disp.push_back(this->operator()(x, y, disp));
      vals_each_disp[disp] = (this->operator()(x, y, disp));
    }
    return vals_each_disp;
  }
  
  /* get the width of an image in the stereo set */
  unsigned int Width() const { return width_; }
  
  /* get the height of an image in the stereo set */
  unsigned int Height() const { return height_; }

  /* get the number of possible disparity values of the stereo set */
  unsigned int NumDisparityVals() const { return num_disp_vals_; }

  /* get the index offset between values corresponding between different
     disparity values for the same pixel */
  unsigned int IndexBtwDispVals() const {
#if defined(INDEXING_W_X_VAL_LAST)
    return width_;
#else
    return 1;
#endif //INDEXING_W_X_VAL_LAST
  }
  
 private:
  //std::vector<T> bp_vals_;
  std::unique_ptr<T[]> bp_vals_;
  unsigned int width_, height_, num_disp_vals_;
};


/* use imRef to access image data. */
#define imRef(im, x, y) (im->access[y][x])
  
/* use imPtr to get pointer to image data. */
#define imPtr(im, x, y) &(im->access[y][x])

template <class T>
image<T>::image(int width, int height, bool init) {
  w = width;
  h = height;
  data = new T[w * h];  // allocate space for image data
  access = new T*[h];   // allocate space for row pointers
  
  // initialize row pointers
  for (int i = 0; i < h; i++)
    access[i] = data + (i * w);  
  
  if (init)
    memset(data, 0, w * h * sizeof(T));
}

template <class T>
image<T>::~image() {
  delete [] data; 
  delete [] access;
}

template <class T>
void image<T>::init(const T &val) {
  T *ptr = imPtr(this, 0, 0);
  T *end = imPtr(this, w-1, h-1);
  while (ptr <= end)
    *ptr++ = val;
}


template <class T>
image<T> *image<T>::copy() const {
  image<T> *im = new image<T>(w, h, false);
  memcpy(im->data, data, w * h * sizeof(T));
  return im;
}

/* use imRef to access image data. */
#define imWDispRef(im, x, y, disp) (im->data[(y * im->width() * im->DisparityVals()) + (x * im->DisparityVals()) + disp])
  
/* use imPtr to get pointer to image data. */
#define imWDispPtr(im, x, y, disp) &(im->data[(y * im->width() * im->DisparityVals()) + (x * im->DisparityVals()) + disp])

template <class T>
imageWDisp<T>::imageWDisp(int width, int height, int num_disp_vals, bool init)
{
  w = width;
  h = height;
  disp_vals = num_disp_vals;
  data = new T[w * h * disp_vals];  // allocate space for image data
  //access = new T*[h];   // allocate space for row pointers
  
  // initialize row pointers
  /*for (int i = 0; i < h; i++)
    access[i] = data + (i * w * disp_vals);*/
  
  if (init)
    memset(data, 0, w * h * disp_vals * sizeof(T));
}

template <class T>
imageWDisp<T>::~imageWDisp() {
  delete [] data; 
  //delete [] access;
}

template <class T>
void imageWDisp<T>::init(const T &val) {
  for (int i=0; i < w*h*disp_vals; i++) {
    data[i] = val;
  }
  /*T *ptr = imPtr(this, 0, 0);
  T *end = imPtr(this, w-1, h-1);
  while (ptr <= end)
    *ptr++ = val;*/
}


template <class T>
imageWDisp<T> *imageWDisp<T>::copy() const {
  imageWDisp<T> *im = new imageWDisp<T>(w, h, disp_vals, false);
  memcpy(im->data, data, w * h * disp_vals * sizeof(T));
  return im;
}

};

#endif
  
