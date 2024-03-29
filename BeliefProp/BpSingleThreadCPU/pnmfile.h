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
const bool USE_WEIGHTED_RGB_TO_GRAYSCALE_CONVERSION_PNMFILE = true;

class pnm_error { };

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
    throw pnm_error();
    
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

static image<uchar> *loadPGM(const char *name) {
  std::string buf;
  
  /* read header */
  std::ifstream file(name, std::ios::in | std::ios::binary);
  pnm_read(file, buf);
  if (strncmp(buf.c_str(), "P5", 2))
    throw pnm_error();

  pnm_read(file, buf);
  int width = std::stoi(buf);
  pnm_read(file, buf);
  int height = std::stoi(buf);

  pnm_read(file, buf);
  if (std::stoi(buf) > UCHAR_MAX)
    throw pnm_error();

  /* read data */
  image<uchar> *im = new image<uchar>(width, height);
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

/*static image<rgb> *loadPPM(const char *name) {
  std::string buf;
  
  //read header
  std::ifstream file(name, std::ios::in | std::ios::binary);
  pnm_read(file, buf);
  if (strncmp(buf.c_str(), "P6", 2))
    throw pnm_error();

  pnm_read(file, buf);
  int width = std::stoi(buf);
  pnm_read(file, buf);
  int height = std::stoi(buf);

  pnm_read(file, buf);
  if (std::stoi(buf) > UCHAR_MAX)
    throw pnm_error();

  //read data
  image<rgb> *im = new image<rgb>(width, height);
  file.read((char *)imPtr(im, 0, 0), width * height * sizeof(rgb));

  return im;
}*/

static image<uchar> *loadPPMAndConvertToGrayScale(const char *name) {
  std::string buf;

  /* read header */
  std::ifstream file(name, std::ios::in | std::ios::binary);
  pnm_read(file, buf);
  if (strncmp(buf.c_str(), "P6", 2))
    throw pnm_error();

  pnm_read(file, buf);
  int width = std::stoi(buf);
  pnm_read(file, buf);
  int height = std::stoi(buf);

  pnm_read(file, buf);
  if (std::stoi(buf) > UCHAR_MAX)
    throw pnm_error();

  /* read data */
  image<rgb> *im = new image<rgb>(width, height);
  image<uchar> *imGrayScale = new image<uchar>(width, height);
  file.read((char *)imPtr(im, 0, 0), width * height * sizeof(rgb));

  float rChannelWeight = 1.0f / 3.0f;
  float gChannelWeight = 1.0f / 3.0f;
  float bChannelWeight = 1.0f / 3.0f;
  if (USE_WEIGHTED_RGB_TO_GRAYSCALE_CONVERSION_PNMFILE)
  {
            rChannelWeight = 0.299f;
            gChannelWeight = 0.114f;
            bChannelWeight = 0.587f;
  }

  for (int i=0; i<width; i++)
  {
    for (int j=0; j<height; j++)
    {
      imRef(imGrayScale, i, j) = floor(((float)imRef(im, i, j).r)*rChannelWeight + ((float)imRef(im, i, j).g)*gChannelWeight + ((float)imRef(im, i, j).b)*bChannelWeight + 0.5f);
    }
  }
  delete im;

  return imGrayScale;
}


/*static void savePPM(image<rgb> *im, const char *name) {
  int width = im->width();
  int height = im->height();
  std::ofstream file(name, std::ios::out | std::ios::binary);

  file << "P6\n" << width << " " << height << "\n" << UCHAR_MAX << "\n";
  file.write((char *)imPtr(im, 0, 0), width * height * sizeof(rgb));
}*/

static image<uchar> *loadPGMOrPPMImage(const char *name) {
  char pgmExtension[] = "pgm";
  char ppmExtension[] = "ppm";
  char* filePathImageCopy = new char[strlen(name) + 1]{};
  std::copy(name, name + strlen(name), filePathImageCopy);

  //check if PGM or PPM image (types currently supported)
#ifdef _WIN32
  char* next_token;
  char* token;
  token = strtok_s(filePathImageCopy, ".", &next_token);
#else
  char* token = strtok(filePathImageCopy, ".");
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
    delete [] filePathImageCopy;

    // load input pgm image
    return loadPGM(name);
  }
  else if (strcmp(lastToken, ppmExtension) == 0)
  {
    delete [] filePathImageCopy;

    // load input ppm image
    return loadPPMAndConvertToGrayScale(name);
  }
  else
  {
    delete [] filePathImageCopy;
    std::cout << "CPU ERROR, IMAGE FILE " << name << " NOT SUPPORTED\n";
    return NULL;
  }
}

template <class T>
void load_image(image<T> **im, const char *name) {
  std::string buf;
  
  /* read header */
  std::ifstream file(name, std::ios::in | std::ios::binary);
  pnm_read(file, buf);
  if (strncmp(buf.c_str(), "VLIB", 9))
    throw pnm_error();

  pnm_read(file, buf);
  int width = std::stoi(buf);
  pnm_read(file, buf);
  int height = std::stoi(buf);

  /* read data */
  *im = new image<T>(width, height);
  file.read((char *)imPtr((*im), 0, 0), width * height * sizeof(T));
}

template <class T>
void save_image(image<T> *im, const char *name) {
  int width = im->width();
  int height = im->height();
  std::ofstream file(name, std::ios::out | std::ios::binary);

  file << "VLIB\n" << width << " " << height << "\n";
  file.write((char *)imPtr(im, 0, 0), width * height * sizeof(T));
}

#endif
