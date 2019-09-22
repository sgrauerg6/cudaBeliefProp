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

#include "stereo.h"

// dt of 1d function
static void dt(float f[NUM_POSSIBLE_DISPARITY_VALUES]) {
	for (int q = 1; q < NUM_POSSIBLE_DISPARITY_VALUES; q++) {
		float prev = f[q - 1] + 1.0F;
		if (prev < f[q])
			f[q] = prev;
	}
	for (int q = NUM_POSSIBLE_DISPARITY_VALUES - 2; q >= 0; q--) {
		float prev = f[q + 1] + 1.0F;
		if (prev < f[q])
			f[q] = prev;
	}
}

// compute message
template<typename T>
void RunBpStereoCPUSingleThread<T>::msg(float s1[NUM_POSSIBLE_DISPARITY_VALUES], float s2[NUM_POSSIBLE_DISPARITY_VALUES], float s3[NUM_POSSIBLE_DISPARITY_VALUES], float s4[NUM_POSSIBLE_DISPARITY_VALUES],
		float dst[NUM_POSSIBLE_DISPARITY_VALUES], float disc_k_bp) {
	float val;

	// aggregate and find min
	float minimum = INF_BP;
	for (int value = 0; value < NUM_POSSIBLE_DISPARITY_VALUES; value++) {
		dst[value] = s1[value] + s2[value] + s3[value] + s4[value];
		if (dst[value] < minimum)
			minimum = dst[value];
	}

	// dt
	dt(dst);

	// truncate
	minimum += disc_k_bp;
	for (int value = 0; value < NUM_POSSIBLE_DISPARITY_VALUES; value++)
		if (minimum < dst[value])
			dst[value] = minimum;

	// normalize
	val = 0;
	for (int value = 0; value < NUM_POSSIBLE_DISPARITY_VALUES; value++)
		val += dst[value];

	val /= NUM_POSSIBLE_DISPARITY_VALUES;
	for (int value = 0; value < NUM_POSSIBLE_DISPARITY_VALUES; value++)
		dst[value] -= val;
}

// computation of data costs
template<typename T>
image<float[NUM_POSSIBLE_DISPARITY_VALUES]> * RunBpStereoCPUSingleThread<T>::comp_data(image<uchar> *img1, image<uchar> *img2, const BPsettings& algSettings) {
	int width = img1->width();
	int height = img1->height();
	image<float[NUM_POSSIBLE_DISPARITY_VALUES]> *data = new image<float[NUM_POSSIBLE_DISPARITY_VALUES]>(width, height);

	image<float> *sm1, *sm2;
	if (algSettings.smoothingSigma >= 0.1) {
		sm1 = FilterImage::smooth(img1, algSettings.smoothingSigma);
		sm2 = FilterImage::smooth(img2, algSettings.smoothingSigma);
	} else {
		sm1 = imageUCHARtoFLOAT(img1);
		sm2 = imageUCHARtoFLOAT(img2);
	}

	for (int y = 0; y < height; y++) {
		for (int x = NUM_POSSIBLE_DISPARITY_VALUES - 1; x < width; x++) {
			for (int value = 0; value < NUM_POSSIBLE_DISPARITY_VALUES; value++) {
				float val = abs(imRef(sm1, x, y) - imRef(sm2, x - value, y));
				imRef(data, x, y)[value] = algSettings.lambda_bp * std::min(val, algSettings.data_k_bp);
			}
		}
	}

	delete sm1;
	delete sm2;
	return data;
}

// generate output from current messages
template<typename T>
image<uchar> * RunBpStereoCPUSingleThread<T>::output(image<float[NUM_POSSIBLE_DISPARITY_VALUES]> *u, image<float[NUM_POSSIBLE_DISPARITY_VALUES]> *d,
		image<float[NUM_POSSIBLE_DISPARITY_VALUES]> *l, image<float[NUM_POSSIBLE_DISPARITY_VALUES]> *r,
		image<float[NUM_POSSIBLE_DISPARITY_VALUES]> *data) {
	int width = data->width();
	int height = data->height();
	image<uchar> *out = new image<uchar>(width, height);

	for (int y = 1; y < height - 1; y++) {
		for (int x = 1; x < width - 1; x++) {
			// keep track of best value for current pixel
			int best = 0;
			float best_val = INF_BP;
			for (int value = 0; value < NUM_POSSIBLE_DISPARITY_VALUES; value++) {
				float val =
				imRef(u, x, y+1)[value] +
				imRef(d, x, y-1)[value] +
				imRef(l, x+1, y)[value] +
				imRef(r, x-1, y)[value] +
				imRef(data, x, y)[value];

				if (val < best_val) {
					best_val = val;
					best = value;
				}
			}
			imRef(out, x, y) = best;
		}
	}

	return out;
}

// belief propagation using checkerboard update scheme
template<typename T>
void RunBpStereoCPUSingleThread<T>::bp_cb(image<float[NUM_POSSIBLE_DISPARITY_VALUES]> *u, image<float[NUM_POSSIBLE_DISPARITY_VALUES]> *d,
		image<float[NUM_POSSIBLE_DISPARITY_VALUES]> *l, image<float[NUM_POSSIBLE_DISPARITY_VALUES]> *r,
		image<float[NUM_POSSIBLE_DISPARITY_VALUES]> *data, int iter, float disc_k_bp) {
	int width = data->width();
	int height = data->height();

	for (int t = 0; t < iter; t++) {
		//std::cout << "iter " << t << "\n";

		for (int y = 1; y < height - 1; y++) {
			for (int x = ((y + t) % 2) + 1; x < width - 1; x += 2) {

				msg(imRef(u, x, y + 1), imRef(l, x + 1, y), imRef(r, x - 1, y),
						imRef(data, x, y), imRef(u, x, y), disc_k_bp);

				msg(imRef(d, x, y - 1), imRef(l, x + 1, y), imRef(r, x - 1, y),
						imRef(data, x, y), imRef(d, x, y), disc_k_bp);

				msg(imRef(u, x, y + 1), imRef(d, x, y - 1), imRef(r, x - 1, y),
						imRef(data, x, y), imRef(r, x, y), disc_k_bp);

				msg(imRef(u, x, y + 1), imRef(d, x, y - 1), imRef(l, x + 1, y),
						imRef(data, x, y), imRef(l, x, y), disc_k_bp);

			}
		}
	}
}

// multiscale belief propagation for image restoration
template<typename T>
image<uchar> * RunBpStereoCPUSingleThread<T>::stereo_ms(image<uchar> *img1, image<uchar> *img2, const BPsettings& algSettings, std::ostream& resultsFile, float& runtime) {
	image<float[NUM_POSSIBLE_DISPARITY_VALUES]> *u[bp_params::LEVELS_BP];
	image<float[NUM_POSSIBLE_DISPARITY_VALUES]> *d[bp_params::LEVELS_BP];
	image<float[NUM_POSSIBLE_DISPARITY_VALUES]> *l[bp_params::LEVELS_BP];
	image<float[NUM_POSSIBLE_DISPARITY_VALUES]> *r[bp_params::LEVELS_BP];
	image<float[NUM_POSSIBLE_DISPARITY_VALUES]> *data[bp_params::LEVELS_BP];

	auto timeStart = std::chrono::system_clock::now();

	// data costs
	data[0] = comp_data(img1, img2, algSettings);

	// data pyramid
	for (int i = 1; i < algSettings.numLevels; i++) {
		int old_width = data[i - 1]->width();
		int old_height = data[i - 1]->height();
		int new_width = (int) ceil(old_width / 2.0);
		int new_height = (int) ceil(old_height / 2.0);

		assert(new_width >= 1);
		assert(new_height >= 1);

		data[i] = new image<float[NUM_POSSIBLE_DISPARITY_VALUES]>(new_width, new_height);
		for (int y = 0; y < old_height; y++) {
			for (int x = 0; x < old_width; x++) {
				for (int value = 0; value < NUM_POSSIBLE_DISPARITY_VALUES; value++) {
					imRef(data[i], x/2, y/2)[value] +=
							imRef(data[i-1], x, y)[value];
				}
			}
		}
	}

	// run bp from coarse to fine
	for (int i = algSettings.numLevels - 1; i >= 0; i--) {
		int width = data[i]->width();
		int height = data[i]->height();

		// allocate & init memory for messages
		if (i == algSettings.numLevels - 1) {
			// in the coarsest level messages are initialized to zero
			u[i] = new image<float[NUM_POSSIBLE_DISPARITY_VALUES]>(width, height);
			d[i] = new image<float[NUM_POSSIBLE_DISPARITY_VALUES]>(width, height);
			l[i] = new image<float[NUM_POSSIBLE_DISPARITY_VALUES]>(width, height);
			r[i] = new image<float[NUM_POSSIBLE_DISPARITY_VALUES]>(width, height);
		} else {
			// initialize messages from values of previous level
			u[i] = new image<float[NUM_POSSIBLE_DISPARITY_VALUES]>(width, height, false);
			d[i] = new image<float[NUM_POSSIBLE_DISPARITY_VALUES]>(width, height, false);
			l[i] = new image<float[NUM_POSSIBLE_DISPARITY_VALUES]>(width, height, false);
			r[i] = new image<float[NUM_POSSIBLE_DISPARITY_VALUES]>(width, height, false);

			for (int y = 0; y < height; y++) {
				for (int x = 0; x < width; x++) {
					for (int value = 0; value < NUM_POSSIBLE_DISPARITY_VALUES; value++) {
						imRef(u[i], x, y)[value] =
								imRef(u[i+1], x/2, y/2)[value];
						imRef(d[i], x, y)[value] =
								imRef(d[i+1], x/2, y/2)[value];
						imRef(l[i], x, y)[value] =
								imRef(l[i+1], x/2, y/2)[value];
						imRef(r[i], x, y)[value] =
								imRef(r[i+1], x/2, y/2)[value];
					}
				}
			}
			// delete old messages and data
			delete u[i + 1];
			delete d[i + 1];
			delete l[i + 1];
			delete r[i + 1];
			delete data[i + 1];
		}

		// BP
		bp_cb(u[i], d[i], l[i], r[i], data[i], algSettings.numIterations, algSettings.disc_k_bp);
	}

	image<uchar> *out = output(u[0], d[0], l[0], r[0], data[0]);

	auto timeEnd = std::chrono::system_clock::now();
	std::chrono::duration<double> diff = timeEnd-timeStart;

	resultsFile << "AVERAGE CPU RUN TIME: " << diff.count() << "\n";
	//std::cout << "CPU RUN TIME: << diff.count() << std::endl;
	runtime = diff.count();

	delete u[0];
	delete d[0];
	delete l[0];
	delete r[0];
	delete data[0];

	return out;
}

template<typename T>
ProcessStereoSetOutput RunBpStereoCPUSingleThread<T>::operator()(const std::string refImagePath, const std::string testImagePath, const BPsettings& algSettings, std::ostream& resultsStream)
{
	image<uchar> *img1, *img2, *out, *edges;

	// load input
	img1 = loadPGMOrPPMImage(refImagePath.c_str());
	img2 = loadPGMOrPPMImage(testImagePath.c_str());
	float runtime = 0.0f;

	// compute disparities
	out = stereo_ms(img1, img2, algSettings, resultsStream, runtime);

	DisparityMap<float> outDispMap(img1->width(), img1->height());

	//set disparity at each point in disparity map
	for (int y = 0; y < img1->height(); y++)
	{
			for (int x = 0; x < img1->width(); x++)
			{
				outDispMap.setPixelAtPoint(x, y, (float)imRef(out, x, y));
			}
	}

	ProcessStereoSetOutput output;
	output.runTime = runtime;
	output.outDisparityMap = std::move(outDispMap);

	delete img1;
	delete img2;
	delete out;

	return output;
}


#if CURRENT_DATA_TYPE_PROCESSING == DATA_TYPE_PROCESSING_FLOAT
template class RunBpStereoCPUSingleThread<float>;
#elif CURRENT_DATA_TYPE_PROCESSING == DATA_TYPE_PROCESSING_DOUBLE
template class RunBpStereoCPUSingleThread<double>;
#elif CURRENT_DATA_TYPE_PROCESSING == DATA_TYPE_PROCESSING_HALF
//float16_t data type used for arm (rather than short)
#ifdef COMPILING_FOR_ARM
template class RunBpStereoCPUSingleThread<float16_t>;
#else
template class RunBpStereoCPUSingleThread<short>;
#endif //COMPILING_FOR_ARM
#endif //CURRENT_DATA_TYPE_PROCESSING == DATA_TYPE_PROCESSING_FLOAT
