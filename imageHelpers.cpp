/*
Copyright (C) 2009 Scott Grauer-Gray, Chandra Kambhamettu, and Kannappan Palaniappan

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

//Defines the functions used to load the input images and store the disparity map image for use in the CUDA BP implementation

#include "imageHelpers.h"

//functions used to load input images/save resulting movment images

#define BUF_SIZE 75
/* read PNM field, skipping comments */
void pnm_read(std::ifstream &file, char *buf) {
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

unsigned int* ImageHelperFunctions::loadImageAsGrayScale(const char* filePathImage, unsigned int& widthImage, unsigned int& heightImage)
{
	char pgmExtension[] = "pgm";
	char ppmExtension[] = "ppm";
	char* filePathImageCopy = new char[strlen(filePathImage) + 1];
	strcpy(filePathImageCopy, filePathImage);

	//check if PGM or PPM image (types currently supported)
	char* token = strtok(filePathImageCopy, ".");
	char* lastToken = new char[strlen(token) + 1];;
	strcpy(lastToken, token);
	while( token != NULL )
	{
		delete [] lastToken;
		lastToken = new char[strlen(token) + 1];
		strcpy(lastToken, token);
	    token = strtok(NULL, ".");
	}

	//last token after "." is file extension
	if (strcmp(lastToken, pgmExtension) == 0)
	{
		delete [] filePathImageCopy;
		return loadImageFromPGM(filePathImage, widthImage, heightImage);
	}
	else if (strcmp(lastToken, ppmExtension) == 0)
	{
		delete [] filePathImageCopy;
		return loadImageFromPPM(filePathImage, widthImage, heightImage);
	}
	else
	{
		delete [] filePathImageCopy;
		printf("ERROR, IMAGE FILE %s NOT SUPPORTED\n", filePathImage);
		return NULL;
	}
}

//load the PGM image and return as an array of floats
unsigned int* ImageHelperFunctions::loadImageFromPGM(const char* filePathPgmImage, unsigned int& widthImage, unsigned int& heightImage)
{
	unsigned int* imageData;

	unsigned char *dataRead;

	pgmRead (filePathPgmImage, &widthImage, &heightImage,
	     dataRead);

	imageData = new unsigned int[widthImage*heightImage];

	for (int numPixel = 0; numPixel < (widthImage*heightImage); numPixel++)
	{
		imageData[numPixel] = (unsigned int)(dataRead[numPixel]);	
	}

	delete [] dataRead;
	return imageData;
}

//load the PPM image, convert to grayscale, and return as an array of floats
unsigned int* ImageHelperFunctions::loadImageFromPPM(const char* filePathPpmImage, unsigned int& widthImage, unsigned int& heightImage)
{
	unsigned int* imageData;
	unsigned char *dataRead;

	ppmReadReturnGrayScale(filePathPpmImage, &widthImage, &heightImage,
	     dataRead, USE_WEIGHTED_RGB_TO_GRAYSCALE_CONVERSION);

	imageData = new unsigned int[widthImage*heightImage];

	for (int numPixel = 0; numPixel < (widthImage*heightImage); numPixel++)
	{
		imageData[numPixel] = (unsigned int)(dataRead[numPixel]);
	}

	delete [] dataRead;
	return imageData;
}


/* INPUT: a filename (char*),row and column dimension variables (long), and
 *   a pointer to a 2D array of unsigned char's of size MAXROWS x MAXCOLS 
 *   (row major).
 * OUTPUT: an integer is returned indicating whether or not the
 *   file was read into memory (in row major order).  1 is returned if the 
 *   file is read correctly, 0 if it is not.  If there are 
 *   too few pixels, the function still returns 1, but returns an error 
 *   message.  Error messages are also returned if a file cannot be open, 
 *   or if the specifications of the file are invalid.
 * NOTE: The case where too many pixels are in a file is not detected.
 */
int ImageHelperFunctions::pgmRead(const char *fileName, unsigned int *cols, unsigned int *rows,
	     unsigned char*& image) {
	  char buf[BUF_SIZE];

	/* read header */
	  std::ifstream file(fileName, std::ios::in | std::ios::binary);
	  pnm_read(file, buf);
	  if (strncmp(buf, "P5", 2))
	    std::cout << "ERROR READING FILE\n";

	  pnm_read(file, buf);
	  *cols = (unsigned int)atoi(buf);
	  pnm_read(file, buf);
	  *rows = (unsigned int)atoi(buf);

	  pnm_read(file, buf);
	  if (atoi(buf) > UCHAR_MAX)
	    std::cout << "ERROR READING FILE\n";

	  image = new unsigned char[(*cols) * (*rows)];

	  /* read data */
	  file.read((char*)image, *cols * *rows * sizeof(char));
	  file.close();

	  return 1;
}

/* INPUT: a filename (char*),row and column dimension variables (long), and
 *   a pointer to a 2D array of unsigned char's of size MAXROWS x MAXCOLS
 *   (row major).
 * OUTPUT: an integer is returned indicating whether or not the
 *   file was read into memory (in row major order).  1 is returned if the
 *   file is read correctly, 0 if it is not.  If there are
 *   too few pixels, the function still returns 1, but returns an error
 *   message.  Error messages are also returned if a file cannot be open,
 *   or if the specifications of the file are invalid.
 * NOTE: The case where too many pixels are in a file is not detected.
 */
int ImageHelperFunctions::ppmReadReturnGrayScale (const char *fileName, unsigned int *cols, unsigned int *rows,
	     unsigned char*& image, bool weightedRGBConversion)
{
  char buf[BUF_SIZE];

	/* read header */
	std::ifstream file(fileName, std::ios::in | std::ios::binary);
	pnm_read(file, buf);
	if (strncmp(buf, "P5", 2))
		std::cout << "ERROR READING FILE\n";

	pnm_read(file, buf);
	*cols = (unsigned int) atoi(buf);
	pnm_read(file, buf);
	*rows = (unsigned int) atoi(buf);

	pnm_read(file, buf);
	if (atoi(buf) > UCHAR_MAX)
		std::cout << "ERROR READING FILE\n";

	image = new unsigned char[(*cols) * (*rows)];
	unsigned char* rgbImage = new unsigned char[3 * (*cols) * (*rows)];

	/* read data */
	file.read((char*) rgbImage, 3 * *cols * *rows * sizeof(char));
	file.close();

	//convert the RGB image to grayscale
	for (int i = 0; i < (*rows) * (*cols); i++) {
		float rChannelWeight = 1.0f / 3.0f;
		float bChannelWeight = 1.0f / 3.0f;
		float gChannelWeight = 1.0f / 3.0f;
		if (weightedRGBConversion) {
			rChannelWeight = 0.299f;
			bChannelWeight = 0.587f;
			gChannelWeight = 0.114f;
		}
		image[i] = (unsigned char) floor(
				rChannelWeight * ((float) rgbImage[i * 3])
						+ gChannelWeight * ((float) rgbImage[i * 3 + 1])
						+ bChannelWeight * ((float) rgbImage[i * 3 + 2])
						+ 0.5f);
	}

	//free memory used for storing rgb image (since using grayscale image)
	delete[] rgbImage;

	return (1);
}

    
/* INPUT: a filename (char*), the dimensions of the pixmap (rows,cols of
 *   type long), and a pointer to a 2D array (MAXROWS x MAXCOLS) in row
 *   major order.
 * OUTPUT: an integer is returned indicating if the desired file was written
 *   (in P5 PGM format (binary)).  A 1 is returned if the write was completed
 *   and 0 if it was not.  An error message is returned if the file is not
 *   properly opened.  
 */ 
int ImageHelperFunctions::pgmWrite(const char* filename, unsigned int cols, unsigned int rows,
	     unsigned char* image,char* comment_string)
{
	  std::ofstream file(filename, std::ios::out | std::ios::binary);

	  file << "P5\n" << cols << " " << rows << "\n" << UCHAR_MAX << "\n";
	  file.write((char*)image, cols * rows * sizeof(char));
	  file.close();
      return(1);
}

//save the calculated disparity map from image 1 to image 2 as a grayscale image using the SCALE_MOVEMENT factor with
//0 representing "zero" intensity and the intensity linearly increasing from there using SCALE_MOVEMENT
void ImageHelperFunctions::saveDisparityImageToPGM(const char* filePathSaveImage, float scaleMovement, const float* calcDisparityBetweenImages, unsigned int widthImage, unsigned int heightImage)
{
	//declare and allocate the space for the movement image to save
	unsigned char* movementImageToSave = new unsigned char[widthImage*heightImage];

	//go though every value in the movementBetweenImages data and retrieve the intensity value to use in the resulting "movement image" where minMovementDirection
	//represents 0 intensity and the intensity increases linearly using scaleMovement from minMovementDirection
	for (unsigned int currentPixel = 0; currentPixel < (widthImage*heightImage); currentPixel++)
	{
		//add .5 and truncate to "round" the intensity to save to an integer
		movementImageToSave[currentPixel] = (unsigned char)((calcDisparityBetweenImages[currentPixel])*scaleMovement + .5f);
	}

	pgmWrite(filePathSaveImage, widthImage, heightImage,
	     movementImageToSave, "blah");
}
