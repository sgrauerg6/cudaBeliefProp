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

#include "imageHelpersHostHeader.cuh"


//functions used to load input images/save resulting movment images

//function to retrieve the disparity values from a disparity map with a known scale factor
float* ImageHelperFunctions::retrieveDisparityValsFromStereoPGM(const char* filePathPgmImage, unsigned int widthImage, unsigned int heightImage, float scaleFactor)
{
	unsigned int* imageData = new unsigned int[widthImage*heightImage];

	float* disparityVals = new float[widthImage*heightImage];

	//go through every pixel and retrieve the Stereo value using the pixel value and the scale factor
	for (unsigned int pixelNum = 0; pixelNum < (widthImage*heightImage); pixelNum++)
	{
		disparityVals[pixelNum] = imageData[pixelNum] / scaleFactor;
	}

	delete [] imageData;

	return disparityVals;
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
		//printf("PGM IMAGE\n");
		return loadImageFromPGM(filePathImage, widthImage, heightImage);
	}
	else if (strcmp(lastToken, ppmExtension) == 0)
	{
		delete [] filePathImageCopy;
		//printf("PPM IMAGE\n");
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
int ImageHelperFunctions::pgmRead (const char *fileName, unsigned int *cols, unsigned int *rows,
	     unsigned char*& image) {
      FILE *filePointer;    /* for file buffer */
      char line[MAXLENGTH]; /* for character input from file */
      int maximumValue = 0; /* max value from header */
      int binary;           /* flag to indicate if file is binary (P5)*/
      long numberRead = 0;  /* counter for number of pixels read */
      long i,j;             /* (i,j) for loops */
      int test,temp;        /* for detecting EOF(test) and temp storage */

      /* Open the file, return an error if necessary. */
      if ((filePointer = fopen(fileName,"r")) == NULL) {
	   printf ("ERROR: cannot open file\n\n");
	   fclose (filePointer);
	   return (0);
      }
    
      /* Initialize columnsidth, and height */
      *cols = *rows =0;

      /* Check the file signature ("Magic Numbers" P2 and P5); skip comments
       * and blank lines (CR with no spaces before it).*/
      fgets (line,MAXLENGTH,filePointer);
      while (line[0]=='#' || line[0]=='\n') fgets (line,MAXLENGTH,filePointer);
      if (line[0]=='P' && (line[1]=='2')) {
	   binary = 0;
	 /*   printf ("\nFile Format: P2\n"); */
      }
      else if (line[0]=='P' && (line[1]=='5')) {
	   binary = 1;
	  /*  printf ("\nFORMAT: P5\n"); */
      }
      else {
	   printf ("ERROR: incorrect file format\n\n");
	   fclose (filePointer);
	   return (0);
      }          

      /* Input the width, height and maximum value, skip comments and blank
       * lines. */
      fgets (line,MAXLENGTH,filePointer);
      while (line[0]=='#' || line[0]=='\n') fgets (line,MAXLENGTH,filePointer);
      sscanf (line,"%u %u",cols,rows);

      fgets (line,MAXLENGTH,filePointer);
      while (line[0]=='#' || line[0]=='\n') fgets(line,MAXLENGTH,filePointer);
      sscanf (line,"%d",&maximumValue);

      /* Check specifications and return an error if h,w, or
      *  maximum value is illegal.*/
      if ((*cols)<1 ||(*rows)<1 || maximumValue<0 || maximumValue>MAXVALUE){
	   printf ("ERROR: invalid file specifications (cols/rows/max value)\n\n");
	   fclose (filePointer);
	   return (0);
      }
      else if ((*cols) > MAXCOLS || (*rows) > MAXROWS) {
	   printf ("ERROR: increase MAXROWS/MAXCOLS in PGM.h");
	   fclose (filePointer);
	   return (0);
      } 

      image = new unsigned char[(*cols)*(*rows)];

      /* Read in the data for binary (P5) or ascii (P2) PGM formats   */
      if (binary) {
	   for (i = 0; i < (*rows); i++) {
	        numberRead += fread((void *)&(image[i*(*cols) + 0]),
		  sizeof(unsigned char), (*cols), filePointer); 
		if (feof(filePointer)) break;
	   }
      }
      else {
	   for (i= 0; i < (*rows); i++) {
	        for (j =0; j < (*cols); j++) { 
	             test = fscanf (filePointer,"%d",&temp);
		     if (test == EOF) break;
		     image[i*(*cols) + j] = (unsigned char)temp;

		     numberRead++;
		}
		if (test == EOF) break;
	   }
      } 
 
      /* Insure the number of pixels read is at least the
       *   number indicated by w*h.
       * If not, return an error message, but proceed */
      if (numberRead < ((*rows)*(*cols))) {
	   printf ("ERROR: fewer pixels than rows*cols indicates\n\n");
      }
     
      /* close the file and return 1 indicating success */
      fclose (filePointer);
      return (1);
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
	     unsigned char*& image, bool weightedRGBConversion) {
      FILE *filePointer;    /* for file buffer */
      char line[MAXLENGTH]; /* for character input from file */
      int maximumValue = 0; /* max value from header */
      int binary;           /* flag to indicate if file is binary (P5)*/
      long numberRead = 0;  /* counter for number of pixels read */
      long i,j;             /* (i,j) for loops */
      int test,temp;        /* for detecting EOF(test) and temp storage */

      /* Open the file, return an error if necessary. */
      if ((filePointer = fopen(fileName,"r")) == NULL) {
	   printf ("ERROR: cannot open file\n\n");
	   fclose (filePointer);
	   return (0);
      }

      /* Initialize columnsidth, and height */
      *cols = *rows =0;

      /* Check the file signature ("Magic Numbers" P2 and P5); skip comments
       * and blank lines (CR with no spaces before it).*/
      fgets (line,MAXLENGTH,filePointer);
      while (line[0]=='#' || line[0]=='\n') fgets (line,MAXLENGTH,filePointer);
      if (line[0]=='P' && (line[1]=='3')) {
	   binary = 0;
	 /*   printf ("\nFile Format: P2\n"); */
      }
      else if (line[0]=='P' && (line[1]=='6')) {
	   binary = 1;
	  /*  printf ("\nFORMAT: P5\n"); */
      }
      else {
	   printf ("ERROR: incorrect file format\n\n");
	   fclose (filePointer);
	   return (0);
      }

      /* Input the width, height and maximum value, skip comments and blank
       * lines. */
      fgets (line,MAXLENGTH,filePointer);
      while (line[0]=='#' || line[0]=='\n') fgets (line,MAXLENGTH,filePointer);
      sscanf (line,"%u %u",cols,rows);

      fgets (line,MAXLENGTH,filePointer);
      while (line[0]=='#' || line[0]=='\n') fgets(line,MAXLENGTH,filePointer);
      sscanf (line,"%d",&maximumValue);

      /* Check specifications and return an error if h,w, or
      *  maximum value is illegal.*/
      if ((*cols)<1 ||(*rows)<1 || maximumValue<0 || maximumValue>MAXVALUE){
	   printf ("ERROR: invalid file specifications (cols/rows/max value)\n\n");
	   fclose (filePointer);
	   return (0);
      }
      else if ((*cols) > MAXCOLS || (*rows) > MAXROWS) {
	   printf ("ERROR: increase MAXROWS/MAXCOLS in PGM.h");
	   fclose (filePointer);
	   return (0);
      }

      unsigned char* rgbImage = new unsigned char[3*(*cols)*(*rows)];
      image = new unsigned char[(*cols)*(*rows)];

      /* Read in the data for binary (P5) or ascii (P2) PGM formats   */
      if (binary) {
	   for (i = 0; i < (*rows); i++) {
	        numberRead += fread((void *)&(rgbImage[(3*i)*(*cols) + 0]),
		  sizeof(unsigned char), 3*(*cols), filePointer);
		if (feof(filePointer)) break;
	   }
      }
      else {
	   for (i= 0; i < (*rows); i++) {
	        for (j =0; j < (3*(*cols)); j++) {
	             test = fscanf (filePointer,"%d",&temp);
	             if (test == EOF) break;
	             rgbImage[i*(*cols) + j] = (unsigned char)temp;
	             numberRead++;
	        }
		if (test == EOF) break;
	   }
      }

      /* Insure the number of pixels read is at least the
       *   number indicated by w*h.
       * If not, return an error message, but proceed */
      if (numberRead < (3*(*rows)*(*cols))) {
	   printf ("ERROR: fewer pixels than rows*cols indicates\n\n");
      }

      //convert the RGB image to grayscale
      for (i = 0; i < (*rows)*(*cols); i++)
      {
    	  float rChannelWeight = 1.0f / 3.0f;
    	  float bChannelWeight = 1.0f / 3.0f;
    	  float gChannelWeight = 1.0f / 3.0f;
    	  if (weightedRGBConversion)
    	  {
    		  rChannelWeight = 0.299f;
    		  bChannelWeight = 0.587f;
    		  gChannelWeight = 0.114f;
    	  }
    	  image[i] = (unsigned char)floor(rChannelWeight*((float)rgbImage[i*3]) + gChannelWeight*((float)rgbImage[i*3 + 1]) + bChannelWeight*((float)rgbImage[i*3 + 2]) + 0.5f);
      }

      //free memory used for storing rgb image (since using grayscale image)
      delete [] rgbImage;

      /* close the file and return 1 indicating success */
      fclose (filePointer);
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
	     unsigned char* image,char* comment_string) {
      FILE* file;        /* pointer to the file buffer */
      //int maxval;        /* maximum value in the image array */
      long nwritten = 0; /* counter for the number of pixels written */
      long i;//,j;          /* for loop counters */

      /* open the file; write header and comments specified by the user. */
      if ((file = fopen(filename, "w")) == NULL)	{
           printf("ERROR: file open failed\n");
	   return(0);
      }
      fprintf(file,"P5\n");

      if (comment_string != NULL) fprintf(file,"# %s \n", comment_string);
    
      /* write the dimensions of the image */	
      fprintf(file,"%ld %ld \n", cols, rows);

      /* NOTE: MAXIMUM VALUE IS WHITE; COLOURS ARE SCALED FROM 0 - */
      /* MAXVALUE IN A .PGM FILE. */
      
      /* WRITE MAXIMUM VALUE TO FILE */
      fprintf(file, "%d\n", (int)255);

      /* Write data */

      for (i=0; i < rows; i++) {
          nwritten += fwrite((void*)&(image[i*cols]),sizeof(unsigned char),
	  		   cols, file);
      }	

      fclose(file);
      return(1);
}

//save the calculated disparity map from image 1 to image 2 as a grayscale image using the SCALE_MOVEMENT factor with
//0 representing "zero" intensity and the intensity linearly increasing from there using SCALE_MOVEMENT
void ImageHelperFunctions::saveDisparityImageToPGM(const char* filePathSaveImage, float scaleMovement, float*& calcDisparityBetweenImages, unsigned int widthImage, unsigned int heightImage)
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

//save the output disparity map using the scale defined in scaleDisparityInOutput at each pixel to the file at disparityMapSaveImagePath
//also takes in the timer to time the implementation including the transfer time from the device to the host
void ImageHelperFunctions::saveResultingDisparityMap(const char* disparityMapSaveImagePath,
		float*& disparityMapFromImage1To2Device, float scaleDisparityInOutput,
		unsigned int widthImages, unsigned int heightImages,
		std::chrono::time_point<std::chrono::system_clock>& timeWithTransferStart,
		double& totalTimeIncludeTransfer) {
	//allocate the space on the host for and x and y movement between images
	float* disparityMapFromImage1To2Host = new float[widthImages * heightImages];

	//transfer the disparity map estimation on the device to the host for output
	(cudaMemcpy(disparityMapFromImage1To2Host, disparityMapFromImage1To2Device, widthImages*heightImages*sizeof(float),
						  cudaMemcpyDeviceToHost) );

	auto timeWithTransferEnd = std::chrono::system_clock::now();

	//printf("Running time including transfer time: %.10lf seconds\n", timeEnd-timeStart);
	std::chrono::duration<double> diff = timeWithTransferEnd-timeWithTransferStart;
	totalTimeIncludeTransfer = diff.count();
	//stop the timer and print the total time of the BP implementation including the device-host transfer time
	//printf("Time to retrieve movement on host (including transfer): %f (ms) \n", totalTimeIncludeTransfer);

	//save the resulting disparity map images to a file
	ImageHelperFunctions::saveDisparityImageToPGM(disparityMapSaveImagePath, scaleDisparityInOutput, disparityMapFromImage1To2Host, widthImages, heightImages);

	delete [] disparityMapFromImage1To2Host;
}



