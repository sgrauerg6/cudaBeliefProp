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

/* read PNM field, skipping comments */
void ImageHelperFunctions::pnm_read(std::ifstream &file, std::string& buf) {
	  std::string doc;
	  char c;

	  file >> c;
	  while (c == '#') {
	    std::getline(file, doc);
	    file >> c;
	  }
	  file.putback(c);

	  file >> buf;
	  file.ignore();
	}


unsigned int* ImageHelperFunctions::loadImageAsGrayScale(const std::string& filePathImage, unsigned int& widthImage, unsigned int& heightImage)
{
	std::string pgmExtension("pgm");
	std::string ppmExtension("ppm");
	std::string filePathImageCopy(filePathImage);

	//check if PGM or PPM image (types currently supported)
	std::istringstream iss(filePathImageCopy);
	std::string token;
	while (std::getline(iss, token, '.'))
	{
		//continue to get last token with the file extension
	}

	//last token after "." is file extension
	//use extension to check if image is pgm or ppm
	if (token == pgmExtension)
	{
		return loadImageFromPGM(filePathImage, widthImage, heightImage);
	}
	else if (token == ppmExtension)
	{
		return loadImageFromPPM(filePathImage, widthImage, heightImage);
	}
	else
	{
		std::cout << "ERROR, IMAGE FILE " << filePathImage << " NOT SUPPORTED\n";
		return nullptr;
	}
}

//load the PGM image and return as an array of floats
unsigned int* ImageHelperFunctions::loadImageFromPGM(const std::string& filePathPgmImage, unsigned int& widthImage, unsigned int& heightImage)
{
	unsigned int* imageData;
	unsigned char* dataRead;

	pgmRead(filePathPgmImage, widthImage, heightImage,
	     dataRead);

	imageData = new unsigned int[widthImage*heightImage];

	//convert each pixel in dataRead to unsigned int and place in imageData array in same location
	std::transform(dataRead, dataRead + (widthImage*heightImage), imageData, [] (const unsigned char i) -> unsigned int { return (unsigned int)i; });

	delete [] dataRead;
	return imageData;
}

//load the PPM image, convert to grayscale, and return as an array of floats
unsigned int* ImageHelperFunctions::loadImageFromPPM(const std::string& filePathPpmImage, unsigned int& widthImage, unsigned int& heightImage)
{
	unsigned int* imageData;
	unsigned char* dataRead;

	ppmReadReturnGrayScale(filePathPpmImage, widthImage, heightImage,
	     dataRead, USE_WEIGHTED_RGB_TO_GRAYSCALE_CONVERSION);

	imageData = new unsigned int[widthImage*heightImage];

	//convert each pixel in dataRead to unsigned int and place in imageData array in same location
	std::transform(dataRead, dataRead + (widthImage*heightImage), imageData, [] (const unsigned char i) -> unsigned int { return (unsigned int)i; });

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
int ImageHelperFunctions::pgmRead(const std::string& fileName, unsigned int& cols, unsigned int& rows,
	     unsigned char*& image)
{
	  std::string buf;

	  /* read header */
	  std::ifstream file(fileName, std::ios::in | std::ios::binary);
	  pnm_read(file, buf);
	  if (buf != "P5")
	    std::cout << "ERROR READING FILE\n";

	  pnm_read(file, buf);
	  cols = std::stoul(buf);
	  pnm_read(file, buf);
	  rows = std::stoul(buf);

	  pnm_read(file, buf);
	  if (std::stoul(buf) > UCHAR_MAX)
	    std::cout << "ERROR READING FILE\n";

	  image = new unsigned char[(cols * rows)];

	  /* read data */
	  file.read((char*)image, (cols * rows * sizeof(char)));
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
int ImageHelperFunctions::ppmReadReturnGrayScale (const std::string& fileName, unsigned int& cols, unsigned int& rows,
	     unsigned char*& image, const bool weightedRGBConversion)
{
	std::string buf;

	/* read header */
	std::ifstream file(fileName, std::ios::in | std::ios::binary);
	pnm_read(file, buf);
	if (buf != "P5")
		std::cout << "ERROR READING FILE\n";

	pnm_read(file, buf);
	cols = std::stoul(buf);
	pnm_read(file, buf);
	rows = std::stoul(buf);

	pnm_read(file, buf);
	if (std::stoul(buf) > UCHAR_MAX)
		std::cout << "ERROR READING FILE\n";

	//std::unique_ptr<char[]> imagePtr(new char[cols*rows]);
	std::unique_ptr<char[]> rgbImagePtr(new char[3*cols*rows]);
	image = new unsigned char[(cols) * (rows)];
	//unsigned char* rgbImage = new unsigned char[3 * (cols) * (rows)];

	/* read data */
	file.read(&(rgbImagePtr[0]), 3 * cols * rows * sizeof(char));
	file.close();

	//convert the RGB image to grayscale
	for (unsigned int i = 0; i < (rows * cols); i++) {
		float rChannelWeight = 1.0f / 3.0f;
		float bChannelWeight = 1.0f / 3.0f;
		float gChannelWeight = 1.0f / 3.0f;
		if (weightedRGBConversion) {
			rChannelWeight = 0.299f;
			bChannelWeight = 0.587f;
			gChannelWeight = 0.114f;
		}
		image[i] = (unsigned char) floor(
				rChannelWeight * ((float) rgbImagePtr[i * 3])
						+ gChannelWeight * ((float) rgbImagePtr[i * 3 + 1])
						+ bChannelWeight * ((float) rgbImagePtr[i * 3 + 2])
						+ 0.5f);
	}

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
int ImageHelperFunctions::pgmWrite(const std::string& filename, const unsigned int cols, const unsigned int rows,
	     unsigned char* image)
{
	  std::ofstream file(filename, std::ios::out | std::ios::binary);

	  file << "P5\n" << cols << " " << rows << "\n" << UCHAR_MAX << "\n";
	  file.write((char*)image, cols * rows * sizeof(char));
	  file.close();
      return(1);
}

//save the calculated disparity map from image 1 to image 2 as a grayscale image using the SCALE_MOVEMENT factor with
//0 representing "zero" intensity and the intensity linearly increasing from there using SCALE_MOVEMENT
void ImageHelperFunctions::saveDisparityImageToPGM(const std::string& filePathSaveImage, const float scaleMovement, const float* calcDisparityBetweenImages, const unsigned int widthImage, const unsigned int heightImage)
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
	     movementImageToSave);
}
