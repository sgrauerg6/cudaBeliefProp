/*
 * stereo.h
 *
 *  Created on: Feb 4, 2017
 *      Author: scottgg
 */

#ifndef STEREO_H_
#define STEREO_H_

#include <cstdio>
#include <iostream>
#include <algorithm>
#include <assert.h>
#include <string>
#include <chrono>
#include <iostream>
#include <array>
#include <utility>
#include "image.h"
#include "misc.h"
#include "pnmfile.h"
#include "filter.h"
#include "imconv.h"
#include "BpRunProcessing/RunBpStereoSet.h"
#include "BpConstsAndParams/bpStereoParameters.h"
#include "BpConstsAndParams/bpStructsAndEnums.h"
#include "BpRunImp/BpParallelParams.h"
#include "RunSettingsEval/RunSettings.h"

template<typename T, unsigned int DISP_VALS>
class RunBpStereoCPUSingleThread : public RunBpStereoSet<T, DISP_VALS, run_environment::AccSetting::NONE>
{
public:
  std::optional<ProcessStereoSetOutput> operator()(const std::array<std::string, 2>& refTestImagePath,
      const beliefprop::BPsettings& algSettings,
      const ParallelParams& parallelParams) override;
  std::string getBpRunDescription() override { return "Single-Thread CPU"; }

private:
  // compute message
  image<float[DISP_VALS]> *comp_data(image<uchar> *img1, image<uchar> *img2, const beliefprop::BPsettings& algSettings);
  void msg(float s1[DISP_VALS], float s2[DISP_VALS], float s3[DISP_VALS], float s4[DISP_VALS],
      float dst[DISP_VALS], const float disc_k_bp);
  void dt(float f[DISP_VALS]);
  image<uchar> *output(image<float[DISP_VALS]> *u, image<float[DISP_VALS]> *d,
      image<float[DISP_VALS]> *l, image<float[DISP_VALS]> *r,
      image<float[DISP_VALS]> *data);
  void bp_cb(image<float[DISP_VALS]> *u, image<float[DISP_VALS]> *d,
      image<float[DISP_VALS]> *l, image<float[DISP_VALS]> *r,
      image<float[DISP_VALS]> *data, const unsigned int iter, const float disc_k_bp);
  std::pair<image<uchar>*, RunData> stereo_ms(image<uchar> *img1, image<uchar> *img2, const beliefprop::BPsettings& algSettings, float& runtime);
};

// dt of 1d function
template<typename T, unsigned int DISP_VALS>
inline void RunBpStereoCPUSingleThread<T, DISP_VALS>::dt(float f[DISP_VALS]) {
  for (unsigned int q = 1; q < DISP_VALS; q++) {
    float prev = f[q - 1] + 1.0F;
    if (prev < f[q])
      f[q] = prev;
  }
  for (int q = (int)DISP_VALS - 2; q >= 0; q--) {
    float prev = f[q + 1] + 1.0F;
    if (prev < f[q])
      f[q] = prev;
  }
}

// compute message
template<typename T, unsigned int DISP_VALS>
inline void RunBpStereoCPUSingleThread<T, DISP_VALS>::msg(float s1[DISP_VALS],
    float s2[DISP_VALS], float s3[DISP_VALS],
    float s4[DISP_VALS], float dst[DISP_VALS],
    const float disc_k_bp) {
  float val;

  // aggregate and find min
  float minimum = bp_consts::INF_BP;
  for (unsigned int value = 0; value < DISP_VALS; value++) {
    dst[value] = s1[value] + s2[value] + s3[value] + s4[value];
    if (dst[value] < minimum)
      minimum = dst[value];
  }

  // dt
  dt(dst);

  // truncate
  minimum += disc_k_bp;
  for (unsigned int value = 0; value < DISP_VALS; value++)
    if (minimum < dst[value])
      dst[value] = minimum;

  // normalize
  val = 0;
  for (unsigned int value = 0; value < DISP_VALS; value++)
    val += dst[value];

  val /= DISP_VALS;
  for (unsigned int value = 0; value < DISP_VALS; value++)
    dst[value] -= val;
}

// computation of data costs
template<typename T, unsigned int DISP_VALS>
inline image<float[DISP_VALS]> * RunBpStereoCPUSingleThread<T, DISP_VALS>::comp_data(
    image<uchar> *img1, image<uchar> *img2, const beliefprop::BPsettings& algSettings) {
  unsigned int width{(unsigned int)img1->width()};
  unsigned int height{(unsigned int)img1->height()};
  image<float[DISP_VALS]> *data = new image<float[DISP_VALS]>(width, height);

  image<float> *sm1, *sm2;
  if (algSettings.smoothingSigma_ >= 0.1) {
    sm1 = FilterImage::smooth(img1, algSettings.smoothingSigma_);
    sm2 = FilterImage::smooth(img2, algSettings.smoothingSigma_);
  } else {
    sm1 = imageUCHARtoFLOAT(img1);
    sm2 = imageUCHARtoFLOAT(img2);
  }

  for (unsigned int y = 0; y < height; y++) {
    for (unsigned int x = DISP_VALS - 1; x < width; x++) {
      for (unsigned int value = 0; value < DISP_VALS; value++) {
        const float val = abs(imRef(sm1, x, y) - imRef(sm2, x - value, y));
        imRef(data, x, y)[value] = algSettings.lambda_bp_ * std::min(val, algSettings.data_k_bp_);
      }
    }
  }

  delete sm1;
  delete sm2;
  return data;
}

// generate output from current messages
template<typename T, unsigned int DISP_VALS>
inline image<uchar> * RunBpStereoCPUSingleThread<T, DISP_VALS>::output(image<float[DISP_VALS]> *u,
    image<float[DISP_VALS]> *d, image<float[DISP_VALS]> *l,
    image<float[DISP_VALS]> *r, image<float[DISP_VALS]> *data) {
  unsigned int width{(unsigned int)data->width()};
  unsigned int height{(unsigned int)data->height()};
  image<uchar> *out = new image<uchar>(width, height);

  for (unsigned int y = 1; y < height - 1; y++) {
    for (unsigned int x = 1; x < width - 1; x++) {
      // keep track of best value for current pixel
      unsigned int best = 0;
      float best_val = bp_consts::INF_BP;
      for (unsigned int value = 0; value < DISP_VALS; value++) {
        const float val =
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
template<typename T, unsigned int DISP_VALS>
inline void RunBpStereoCPUSingleThread<T, DISP_VALS>::bp_cb(image<float[DISP_VALS]> *u, image<float[DISP_VALS]> *d,
    image<float[DISP_VALS]> *l, image<float[DISP_VALS]> *r,
    image<float[DISP_VALS]> *data, const unsigned int iter, const float disc_k_bp) {
  unsigned int width{(unsigned int)data->width()};
  unsigned int height{(unsigned int)data->height()};

  for (unsigned int t = 0; t < iter; t++) {
    //std::cout << "iter " << t << "\n";

    for (unsigned int y = 1; y < height - 1; y++) {
      for (unsigned int x = ((y + t) % 2) + 1; x < width - 1; x += 2) {

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
template<typename T, unsigned int DISP_VALS>
inline std::pair<image<uchar>*, RunData> RunBpStereoCPUSingleThread<T, DISP_VALS>::stereo_ms(image<uchar> *img1, image<uchar> *img2,
  const beliefprop::BPsettings& algSettings, float& runtime) {
  image<float[DISP_VALS]> *u[bp_params::LEVELS_BP];
  image<float[DISP_VALS]> *d[bp_params::LEVELS_BP];
  image<float[DISP_VALS]> *l[bp_params::LEVELS_BP];
  image<float[DISP_VALS]> *r[bp_params::LEVELS_BP];
  image<float[DISP_VALS]> *data[bp_params::LEVELS_BP];

  auto timeStart = std::chrono::system_clock::now();

  // data costs
  data[0] = comp_data(img1, img2, algSettings);

  // data pyramid
  for (unsigned int i = 1; i < algSettings.numLevels_; i++) {
    const unsigned int old_width = (unsigned int)data[i - 1]->width();
    const unsigned int old_height = (unsigned int)data[i - 1]->height();
    const unsigned int new_width = (unsigned int) ceil(old_width / 2.0);
    const unsigned int new_height = (unsigned int) ceil(old_height / 2.0);

    assert(new_width >= 1);
    assert(new_height >= 1);

    data[i] = new image<float[DISP_VALS]>(new_width, new_height);
    for (unsigned int y = 0; y < old_height; y++) {
      for (unsigned int x = 0; x < old_width; x++) {
        for (unsigned int value = 0; value < DISP_VALS; value++) {
          imRef(data[i], x/2, y/2)[value] +=
              imRef(data[i-1], x, y)[value];
        }
      }
    }
  }

  // run bp from coarse to fine
  for (int i = algSettings.numLevels_ - 1; i >= 0; i--) {
    unsigned int width = (unsigned int)data[i]->width();
    unsigned int height = (unsigned int)data[i]->height();

    // allocate & init memory for messages
    if ((unsigned int)i == (algSettings.numLevels_ - 1)) {
      // in the coarsest level messages are initialized to zero
      u[i] = new image<float[DISP_VALS]>(width, height);
      d[i] = new image<float[DISP_VALS]>(width, height);
      l[i] = new image<float[DISP_VALS]>(width, height);
      r[i] = new image<float[DISP_VALS]>(width, height);
    } else {
      // initialize messages from values of previous level
      u[i] = new image<float[DISP_VALS]>(width, height, false);
      d[i] = new image<float[DISP_VALS]>(width, height, false);
      l[i] = new image<float[DISP_VALS]>(width, height, false);
      r[i] = new image<float[DISP_VALS]>(width, height, false);

      for (unsigned int y = 0; y < height; y++) {
        for (unsigned int x = 0; x < width; x++) {
          for (unsigned int value = 0; value < DISP_VALS; value++) {
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
    bp_cb(u[i], d[i], l[i], r[i], data[i], algSettings.numIterations_, algSettings.disc_k_bp_);
  }

  image<uchar> *out = output(u[0], d[0], l[0], r[0], data[0]);

  auto timeEnd = std::chrono::system_clock::now();
  std::chrono::duration<double> diff = timeEnd-timeStart;

  RunData runData;
  runData.addDataWHeader("AVERAGE CPU RUN TIME", std::to_string(diff.count()));

  delete u[0];
  delete d[0];
  delete l[0];
  delete r[0];
  delete data[0];

  return {out, runData};
}

template<typename T, unsigned int DISP_VALS>
inline std::optional<ProcessStereoSetOutput> RunBpStereoCPUSingleThread<T, DISP_VALS>::operator()(const std::array<std::string, 2>& refTestImagePath,
    const beliefprop::BPsettings& algSettings, const ParallelParams& parallelParams)
{
  image<uchar> *img1, *img2, *out;// *edges;

  // load input
  img1 = loadPGMOrPPMImage(refTestImagePath[0].c_str());
  img2 = loadPGMOrPPMImage(refTestImagePath[1].c_str());
  float runtime = 0.0f;

  // compute disparities
  auto outStereo = stereo_ms(img1, img2, algSettings, runtime);
  out = outStereo.first;

  DisparityMap<float> outDispMap(std::array<unsigned int, 2>{(unsigned int)img1->width(), (unsigned int)img1->height()});

  //set disparity at each point in disparity map
  for (unsigned int y = 0; y < (unsigned int)img1->height(); y++) {
    for (unsigned int x = 0; x < (unsigned int)img1->width(); x++) {
      outDispMap.setPixelAtPoint({x, y}, (float)imRef(out, x, y));
    }
  }

  std::optional<ProcessStereoSetOutput> output{ProcessStereoSetOutput{}};
  output->runTime = runtime;
  output->outDisparityMap = std::move(outDispMap);
  output->runData = outStereo.second;

  delete img1;
  delete img2;
  delete out;

  return output;
}

#ifdef _WIN32

extern "C" __declspec(dllexport) RunBpStereoSet<float, bp_params::STEREO_SETS_TO_PROCESS[0].numDispVals_>* __cdecl createRunBpStereoCPUSingleThreadFloat_KnownDisp0();
extern "C" __declspec(dllexport) RunBpStereoSet<double, bp_params::STEREO_SETS_TO_PROCESS[0].numDispVals_>* __cdecl createRunBpStereoCPUSingleThreadDouble_KnownDisp0();
extern "C" __declspec(dllexport) RunBpStereoSet<short, bp_params::STEREO_SETS_TO_PROCESS[0].numDispVals_>* __cdecl createRunBpStereoCPUSingleThreadShort_KnownDisp0();
extern "C" __declspec(dllexport) RunBpStereoSet<float, bp_params::STEREO_SETS_TO_PROCESS[1].numDispVals_> * __cdecl createRunBpStereoCPUSingleThreadFloat_KnownDisp1();
extern "C" __declspec(dllexport) RunBpStereoSet<double, bp_params::STEREO_SETS_TO_PROCESS[1].numDispVals_> * __cdecl createRunBpStereoCPUSingleThreadDouble_KnownDisp1();
extern "C" __declspec(dllexport) RunBpStereoSet<short, bp_params::STEREO_SETS_TO_PROCESS[1].numDispVals_> * __cdecl createRunBpStereoCPUSingleThreadShort_KnownDisp1();
extern "C" __declspec(dllexport) RunBpStereoSet<float, bp_params::STEREO_SETS_TO_PROCESS[2].numDispVals_> * __cdecl createRunBpStereoCPUSingleThreadFloat_KnownDisp2();
extern "C" __declspec(dllexport) RunBpStereoSet<double, bp_params::STEREO_SETS_TO_PROCESS[2].numDispVals_> * __cdecl createRunBpStereoCPUSingleThreadDouble_KnownDisp2();
extern "C" __declspec(dllexport) RunBpStereoSet<short, bp_params::STEREO_SETS_TO_PROCESS[2].numDispVals_> * __cdecl createRunBpStereoCPUSingleThreadShort_KnownDisp2();
extern "C" __declspec(dllexport) RunBpStereoSet<float, bp_params::STEREO_SETS_TO_PROCESS[3].numDispVals_> * __cdecl createRunBpStereoCPUSingleThreadFloat_KnownDisp3();
extern "C" __declspec(dllexport) RunBpStereoSet<double, bp_params::STEREO_SETS_TO_PROCESS[3].numDispVals_> * __cdecl createRunBpStereoCPUSingleThreadDouble_KnownDisp3();
extern "C" __declspec(dllexport) RunBpStereoSet<short, bp_params::STEREO_SETS_TO_PROCESS[3].numDispVals_> * __cdecl createRunBpStereoCPUSingleThreadShort_KnownDisp3();
extern "C" __declspec(dllexport) RunBpStereoSet<float, bp_params::STEREO_SETS_TO_PROCESS[4].numDispVals_> * __cdecl createRunBpStereoCPUSingleThreadFloat_KnownDisp4();
extern "C" __declspec(dllexport) RunBpStereoSet<double, bp_params::STEREO_SETS_TO_PROCESS[4].numDispVals_> * __cdecl createRunBpStereoCPUSingleThreadDouble_KnownDisp4();
extern "C" __declspec(dllexport) RunBpStereoSet<short, bp_params::STEREO_SETS_TO_PROCESS[4].numDispVals_> * __cdecl createRunBpStereoCPUSingleThreadShort_KnownDisp4();
extern "C" __declspec(dllexport) RunBpStereoSet<float, bp_params::STEREO_SETS_TO_PROCESS[5].numDispVals_> * __cdecl createRunBpStereoCPUSingleThreadFloat_KnownDisp5();
extern "C" __declspec(dllexport) RunBpStereoSet<double, bp_params::STEREO_SETS_TO_PROCESS[5].numDispVals_> * __cdecl createRunBpStereoCPUSingleThreadDouble_KnownDisp5();
extern "C" __declspec(dllexport) RunBpStereoSet<short, bp_params::STEREO_SETS_TO_PROCESS[5].numDispVals_> * __cdecl createRunBpStereoCPUSingleThreadShort_KnownDisp5(); 
extern "C" __declspec(dllexport) RunBpStereoSet<float, bp_params::STEREO_SETS_TO_PROCESS[6].numDispVals_> * __cdecl createRunBpStereoCPUSingleThreadFloat_KnownDisp6();
extern "C" __declspec(dllexport) RunBpStereoSet<double, bp_params::STEREO_SETS_TO_PROCESS[6].numDispVals_> * __cdecl createRunBpStereoCPUSingleThreadDouble_KnownDisp6();
extern "C" __declspec(dllexport) RunBpStereoSet<short, bp_params::STEREO_SETS_TO_PROCESS[6].numDispVals_> * __cdecl createRunBpStereoCPUSingleThreadShort_KnownDisp6();

#endif //_WIN32

#endif /* STEREO_H_ */
