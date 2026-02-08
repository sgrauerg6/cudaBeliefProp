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

/* basic image I/O */

#ifndef PNM_FILE_H
#define PNM_FILE_H

#include <cstdlib>
#include <climits>
#include <cstring>
#include <fstream>
#include "image.h"
#include "misc.h"

#define BUF_SIZE 256
const bool kUseWeightedRGBToGrayscaleConversion_PNMFILE = true;

/**
 * @brief Class and structs in single-thread CPU bp implementation by Pedro
 * Felzenwalb available at https://cs.brown.edu/people/pfelzens/bp/index.html
 */
namespace bp_single_thread_imp {
class pnm_error { };
}
/*static void read_packed(unsigned char *data, int size, std::ifstream &f) {
  unsigned char c = 0;
  
  int bitshift = -1;
  for (int pos = 0; pos < size; pos++) {
    if (bitshift == -1) {
      c = f.get();
      bitshift = 7;
    }
    data[pos] = (c >> bitshift) & 1;
    bitshift--;
    }
}*/

/*static void write_packed(unsigned char *data, int size, std::ofstream &f) {
  unsigned char c = 0;
  
  int bitshift = 7;
  for (int pos = 0; pos < size; pos++) {
      c = c | (data[pos] << bitshift);
      bitshift--;
      if ((bitshift == -1) || (pos == size-1)) {
  f.put(c);
  bitshift = 7;
  c = 0;
      }
  }
}*/

/* read PNM field, skipping comments */ 
static void pnm_read(std::ifstream &file, std::string& buf) {
  char doc[BUF_SIZE];
  char c;
  
  file >> c;
  while (c == '#') {
    file.getline(doc, BUF_SIZE);
    file >> c;
  }
  file.putback(c);
  
  file.width(BUF_SIZE);
  file >> buf;
  file.ignore();
}

/*static image<uchar> *loadPBM(const char *name) {
  std::string buf;
  
  //read header
  std::ifstream file(name, std::ios::in | std::ios::binary);
  pnm_read(file, buf);
  if (strncmp(buf.c_str(), "P4", 2))
    throw bp_single_thread_imp::pnm_error();
    
  pnm_read(file, buf);
  int width = std::stoi(buf);
  pnm_read(file, buf);
  int height = std::stoi(buf);
  
  //read data
  image<uchar> *im = new image<uchar>(width, height);
  for (int i = 0; i < height; i++)
    read_packed(imPtr(im, 0, i), width, file);
  
  return im;
}*/

/*static void savePBM(image<uchar> *im, const char *name) {
  int width = im->width();
  int height = im->height();
  std::ofstream file(name, std::ios::out | std::ios::binary);

  file << "P4\n" << width << " " << height << "\n";
  for (int i = 0; i < height; i++)
    write_packed(imPtr(im, 0, i), width, file);
}*/

static bp_single_thread_imp::image<uchar> *loadPGM(const char *name) {
  std::string buf;
  
  /* read header */
  std::ifstream file(name, std::ios::in | std::ios::binary);
  pnm_read(file, buf);
  if (strncmp(buf.c_str(), "P5", 2))
    throw bp_single_thread_imp::pnm_error();

  pnm_read(file, buf);
  int width = std::stoi(buf);
  pnm_read(file, buf);
  int height = std::stoi(buf);

  pnm_read(file, buf);
  if (std::stoi(buf) > UCHAR_MAX)
    throw bp_single_thread_imp::pnm_error();

  /* read data */
  bp_single_thread_imp::image<uchar> *im = new bp_single_thread_imp::image<uchar>(width, height);
  file.read((char *)imPtr(im, 0, 0), width * height * sizeof(uchar));

  return im;
}

/*static void savePGM(image<uchar> *im, const char *name) {
  int width = im->width();
  int height = im->height();
  std::ofstream file(name, std::ios::out | std::ios::binary);

  file << "P5\n" << width << " " << height << "\n" << UCHAR_MAX << "\n";
  file.write((char *)imPtr(im, 0, 0), width * height * sizeof(uchar));
}*/

/*static image<bp_single_thread_imp::rgb> *loadPPM(const char *name) {
  std::string buf;
  
  //read header
  std::ifstream file(name, std::ios::in | std::ios::binary);
  pnm_read(file, buf);
  if (strncmp(buf.c_str(), "P6", 2))
    throw bp_single_thread_imp::pnm_error();

  pnm_read(file, buf);
  int width = std::stoi(buf);
  pnm_read(file, buf);
  int height = std::stoi(buf);

  pnm_read(file, buf);
  if (std::stoi(buf) > UCHAR_MAX)
    throw bp_single_thread_imp::pnm_error();

  //read data
  image<bp_single_thread_imp::rgb> *im = new image<bp_single_thread_imp::rgb>(width, height);
  file.read((char *)imPtr(im, 0, 0), width * height * sizeof(bp_single_thread_imp::rgb));

  return im;
}*/

static bp_single_thread_imp::image<uchar> *loadPPMAndConvertToGrayScale(const char *name) {
  std::string buf;

  /* read header */
  std::ifstream file(name, std::ios::in | std::ios::binary);
  pnm_read(file, buf);
  if (strncmp(buf.c_str(), "P6", 2))
    throw bp_single_thread_imp::pnm_error();

  pnm_read(file, buf);
  int width = std::stoi(buf);
  pnm_read(file, buf);
  int height = std::stoi(buf);

  pnm_read(file, buf);
  if (std::stoi(buf) > UCHAR_MAX)
    throw bp_single_thread_imp::pnm_error();

  /* read data */
  bp_single_thread_imp::image<bp_single_thread_imp::rgb> *im = new bp_single_thread_imp::image<bp_single_thread_imp::rgb>(width, height);
  bp_single_thread_imp::image<uchar> *imGrayScale = new bp_single_thread_imp::image<uchar>(width, height);
  file.read((char *)imPtr(im, 0, 0), width * height * sizeof(bp_single_thread_imp::rgb));

  float r_channel_weight = 1.0f / 3.0f;
  float g_channel_weight = 1.0f / 3.0f;
  float b_channel_weight = 1.0f / 3.0f;
  if (kUseWeightedRGBToGrayscaleConversion_PNMFILE)
  {
            r_channel_weight = 0.299f;
            g_channel_weight = 0.114f;
            b_channel_weight = 0.587f;
  }

  for (int i=0; i<width; i++)
  {
    for (int j=0; j<height; j++)
    {
      imRef(imGrayScale, i, j) = floor(((float)imRef(im, i, j).r)*r_channel_weight + ((float)imRef(im, i, j).g)*g_channel_weight + ((float)imRef(im, i, j).b)*b_channel_weight + 0.5f);
    }
  }
  delete im;

  return imGrayScale;
}


/*static void savePPM(image<bp_single_thread_imp::rgb> *im, const char *name) {
  int width = im->width();
  int height = im->height();
  std::ofstream file(name, std::ios::out | std::ios::binary);

  file << "P6\n" << width << " " << height << "\n" << UCHAR_MAX << "\n";
  file.write((char *)imPtr(im, 0, 0), width * height * sizeof(bp_single_thread_imp::rgb));
}*/

static inline bp_single_thread_imp::image<uchar> *loadPGMOrPPMImage(const char *name) {
  char pgmExtension[] = "pgm";
  char ppmExtension[] = "ppm";
  char* file_path_image_copy = new char[strlen(name) + 1]{};
  std::copy(name, name + strlen(name), file_path_image_copy);

  //check if PGM or PPM image (types currently supported)
#ifdef _WIN32
  char* next_token;
  char* token;
  token = strtok_s(file_path_image_copy, ".", &next_token);
#else
  char* token = strtok(file_path_image_copy, ".");
#endif //_WIN32
  char* lastToken = new char[strlen(token) + 1]{};
  std::copy(token, token + strlen(token), lastToken);
  while( token != NULL )
  {
    delete [] lastToken;
    lastToken = new char[strlen(token) + 1]{};
    std::copy(token, token + strlen(token), lastToken);
#ifdef _WIN32
    token = strtok_s(NULL, ".", &next_token);
#else
    token = strtok(NULL, ".");
#endif //_WIN32
  }

  //last token after "." is file extension
  if (strcmp(lastToken, pgmExtension) == 0)
  {
    delete [] file_path_image_copy;

    // load input pgm image
    return loadPGM(name);
  }
  else if (strcmp(lastToken, ppmExtension) == 0)
  {
    delete [] file_path_image_copy;

    // load input ppm image
    return loadPPMAndConvertToGrayScale(name);
  }
  else
  {
    delete [] file_path_image_copy;
    std::cout << "CPU ERROR, IMAGE FILE " << name << " NOT SUPPORTED\n";
    return NULL;
  }
}

template <class T>
void load_image(bp_single_thread_imp::image<T> **im, const char *name) {
  std::string buf;
  
  /* read header */
  std::ifstream file(name, std::ios::in | std::ios::binary);
  pnm_read(file, buf);
  if (strncmp(buf.c_str(), "VLIB", 9))
    throw bp_single_thread_imp::pnm_error();

  pnm_read(file, buf);
  int width = std::stoi(buf);
  pnm_read(file, buf);
  int height = std::stoi(buf);

  /* read data */
  *im = new bp_single_thread_imp::image<T>(width, height);
  file.read((char *)imPtr((*im), 0, 0), width * height * sizeof(T));
}

template <class T>
void save_image(bp_single_thread_imp::image<T> *im, const char *name) {
  int width = im->width();
  int height = im->height();
  std::ofstream file(name, std::ios::out | std::ios::binary);

  file << "VLIB\n" << width << " " << height << "\n";
  file.write((char *)imPtr(im, 0, 0), width * height * sizeof(T));
}

#endif
