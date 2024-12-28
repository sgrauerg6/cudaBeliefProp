/*
Copyright (C) 2024 Scott Grauer-Gray

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

/**
 * @file stereo.h
 * @author Scott Grauer-Gray
 * @brief 
 * 
 * @copyright Copyright (c) 2024
 */
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
#include "BpRunProcessing/ParallelParamsBp.h"
#include "BpRunProcessing/RunBpOnStereoSet.h"
#include "BpResultsEvaluation/BpEvaluationStereoSets.h"
#include "BpRunProcessing/BpConstsEnumsAliases.h"
#include "RunSettingsParams/RunSettings.h"

/**
 * @brief Child class of RunBpOnStereoSet to run single-threaded CPU implementation of belief propagation on a
 * given stereo set as defined by reference and test image file paths
 * 
 * @tparam T 
 * @tparam DISP_VALS 
 * @tparam ACCELERATION 
 */
template<typename T, unsigned int DISP_VALS, run_environment::AccSetting ACCELERATION>
class RunBpOnStereoSetSingleThreadCPU final : public RunBpOnStereoSet<T, DISP_VALS, ACCELERATION>
{
public:
  std::optional<beliefprop::BpRunOutput> operator()(const std::array<std::string, 2>& ref_test_image_path,
      const beliefprop::BpSettings& alg_settings,
      const ParallelParams& parallel_params) const override;
  std::string BpRunDescription() const override { return "Single-Thread CPU"; }

private:
  // compute message
  bp_single_thread_imp::image<float[DISP_VALS]> *comp_data(bp_single_thread_imp::image<uchar> *img1, bp_single_thread_imp::image<uchar> *img2, const beliefprop::BpSettings& alg_settings) const;
  void msg(float s1[DISP_VALS], float s2[DISP_VALS], float s3[DISP_VALS], float s4[DISP_VALS],
      float dst[DISP_VALS], float disc_k_bp) const;
  void dt(float f[DISP_VALS]) const;
  bp_single_thread_imp::image<uchar> *output(bp_single_thread_imp::image<float[DISP_VALS]> *u, bp_single_thread_imp::image<float[DISP_VALS]> *d,
      bp_single_thread_imp::image<float[DISP_VALS]> *l, bp_single_thread_imp::image<float[DISP_VALS]> *r,
      bp_single_thread_imp::image<float[DISP_VALS]> *data) const;
  void bp_cb(bp_single_thread_imp::image<float[DISP_VALS]> *u, bp_single_thread_imp::image<float[DISP_VALS]> *d,
      bp_single_thread_imp::image<float[DISP_VALS]> *l, bp_single_thread_imp::image<float[DISP_VALS]> *r,
      bp_single_thread_imp::image<float[DISP_VALS]> *data, unsigned int iter, float disc_k_bp) const;
  std::pair<bp_single_thread_imp::image<uchar>*, RunData> stereo_ms(bp_single_thread_imp::image<uchar> *img1, bp_single_thread_imp::image<uchar> *img2,
    const beliefprop::BpSettings& alg_settings, std::chrono::duration<double>& runtime) const;
};

// dt of 1d function
template<typename T, unsigned int DISP_VALS, run_environment::AccSetting ACCELERATION>
inline void RunBpOnStereoSetSingleThreadCPU<T, DISP_VALS, ACCELERATION>::dt(float f[DISP_VALS]) const {
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
template<typename T, unsigned int DISP_VALS, run_environment::AccSetting ACCELERATION>
inline void RunBpOnStereoSetSingleThreadCPU<T, DISP_VALS, ACCELERATION>::msg(float s1[DISP_VALS],
    float s2[DISP_VALS], float s3[DISP_VALS],
    float s4[DISP_VALS], float dst[DISP_VALS],
    float disc_k_bp) const {
  float val;

  // aggregate and find min
  float minimum = beliefprop::kHighValBp<float>;
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
template<typename T, unsigned int DISP_VALS, run_environment::AccSetting ACCELERATION>
inline bp_single_thread_imp::image<float[DISP_VALS]> * RunBpOnStereoSetSingleThreadCPU<T, DISP_VALS, ACCELERATION>::comp_data(
    bp_single_thread_imp::image<uchar> *img1, bp_single_thread_imp::image<uchar> *img2, const beliefprop::BpSettings& alg_settings) const {
  unsigned int width{(unsigned int)img1->width()};
  unsigned int height{(unsigned int)img1->height()};
  bp_single_thread_imp::image<float[DISP_VALS]> *data = new bp_single_thread_imp::image<float[DISP_VALS]>(width, height);

  bp_single_thread_imp::image<float> *sm1, *sm2;
  if (alg_settings.smoothing_sigma >= 0.1) {
    sm1 = bp_single_thread_imp::FilterImage::smooth(img1, alg_settings.smoothing_sigma);
    sm2 = bp_single_thread_imp::FilterImage::smooth(img2, alg_settings.smoothing_sigma);
  } else {
    sm1 = imageUCHARtoFLOAT(img1);
    sm2 = imageUCHARtoFLOAT(img2);
  }

  for (unsigned int y = 0; y < height; y++) {
    for (unsigned int x = DISP_VALS - 1; x < width; x++) {
      for (unsigned int value = 0; value < DISP_VALS; value++) {
        const float val = abs(imRef(sm1, x, y) - imRef(sm2, x - value, y));
        imRef(data, x, y)[value] = alg_settings.lambda_bp * std::min(val, alg_settings.data_k_bp);
      }
    }
  }

  delete sm1;
  delete sm2;
  return data;
}

// generate output from current messages
template<typename T, unsigned int DISP_VALS, run_environment::AccSetting ACCELERATION>
inline bp_single_thread_imp::image<uchar> * RunBpOnStereoSetSingleThreadCPU<T, DISP_VALS, ACCELERATION>::output(bp_single_thread_imp::image<float[DISP_VALS]> *u,
    bp_single_thread_imp::image<float[DISP_VALS]> *d, bp_single_thread_imp::image<float[DISP_VALS]> *l,
    bp_single_thread_imp::image<float[DISP_VALS]> *r, bp_single_thread_imp::image<float[DISP_VALS]> *data) const {
  unsigned int width{(unsigned int)data->width()};
  unsigned int height{(unsigned int)data->height()};
  bp_single_thread_imp::image<uchar> *out = new bp_single_thread_imp::image<uchar>(width, height);

  for (unsigned int y = 1; y < height - 1; y++) {
    for (unsigned int x = 1; x < width - 1; x++) {
      // keep track of best value for current pixel
      unsigned int best = 0;
      float best_val = beliefprop::kHighValBp<float>;
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
template<typename T, unsigned int DISP_VALS, run_environment::AccSetting ACCELERATION>
inline void RunBpOnStereoSetSingleThreadCPU<T, DISP_VALS, ACCELERATION>::bp_cb(bp_single_thread_imp::image<float[DISP_VALS]> *u, bp_single_thread_imp::image<float[DISP_VALS]> *d,
    bp_single_thread_imp::image<float[DISP_VALS]> *l, bp_single_thread_imp::image<float[DISP_VALS]> *r,
    bp_single_thread_imp::image<float[DISP_VALS]> *data, unsigned int iter, float disc_k_bp) const {
  unsigned int width{(unsigned int)data->width()};
  unsigned int height{(unsigned int)data->height()};

  for (unsigned int t = 0; t < iter; t++) {
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
template<typename T, unsigned int DISP_VALS, run_environment::AccSetting ACCELERATION>
inline std::pair<bp_single_thread_imp::image<uchar>*, RunData> RunBpOnStereoSetSingleThreadCPU<T, DISP_VALS, ACCELERATION>::stereo_ms(bp_single_thread_imp::image<uchar> *img1, bp_single_thread_imp::image<uchar> *img2,
  const beliefprop::BpSettings& alg_settings, std::chrono::duration<double>& runtime) const {
  bp_single_thread_imp::image<float[DISP_VALS]> *u[alg_settings.num_levels];
  bp_single_thread_imp::image<float[DISP_VALS]> *d[alg_settings.num_levels];
  bp_single_thread_imp::image<float[DISP_VALS]> *l[alg_settings.num_levels];
  bp_single_thread_imp::image<float[DISP_VALS]> *r[alg_settings.num_levels];
  bp_single_thread_imp::image<float[DISP_VALS]> *data[alg_settings.num_levels];

  auto timeStart = std::chrono::system_clock::now();

  // data costs
  data[0] = comp_data(img1, img2, alg_settings);

  // data pyramid
  for (unsigned int i = 1; i < alg_settings.num_levels; i++) {
    const unsigned int old_width = (unsigned int)data[i - 1]->width();
    const unsigned int old_height = (unsigned int)data[i - 1]->height();
    const unsigned int new_width = (unsigned int)ceil(old_width / 2.0);
    const unsigned int new_height = (unsigned int)ceil(old_height / 2.0);

    assert(new_width >= 1);
    assert(new_height >= 1);

    data[i] = new bp_single_thread_imp::image<float[DISP_VALS]>(new_width, new_height);
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
  for (int i = alg_settings.num_levels - 1; i >= 0; i--) {
    unsigned int width = (unsigned int)data[i]->width();
    unsigned int height = (unsigned int)data[i]->height();

    // allocate & init memory for messages
    if ((unsigned int)i == (alg_settings.num_levels - 1)) {
      // in the coarsest level messages are initialized to zero
      u[i] = new bp_single_thread_imp::image<float[DISP_VALS]>(width, height);
      d[i] = new bp_single_thread_imp::image<float[DISP_VALS]>(width, height);
      l[i] = new bp_single_thread_imp::image<float[DISP_VALS]>(width, height);
      r[i] = new bp_single_thread_imp::image<float[DISP_VALS]>(width, height);
    } else {
      // initialize messages from values of previous level
      u[i] = new bp_single_thread_imp::image<float[DISP_VALS]>(width, height, false);
      d[i] = new bp_single_thread_imp::image<float[DISP_VALS]>(width, height, false);
      l[i] = new bp_single_thread_imp::image<float[DISP_VALS]>(width, height, false);
      r[i] = new bp_single_thread_imp::image<float[DISP_VALS]>(width, height, false);

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
    bp_cb(u[i], d[i], l[i], r[i], data[i], alg_settings.num_iterations, alg_settings.disc_k_bp);
  }

  bp_single_thread_imp::image<uchar> *out = output(u[0], d[0], l[0], r[0], data[0]);

  auto timeEnd = std::chrono::system_clock::now();
  runtime = timeEnd-timeStart;
  
  RunData run_data;
  run_data.AddDataWHeader(std::string(run_eval::kSingleThreadRuntimeHeader), runtime.count());

  delete u[0];
  delete d[0];
  delete l[0];
  delete r[0];
  delete data[0];

  return {out, run_data};
}

template<typename T, unsigned int DISP_VALS, run_environment::AccSetting ACCELERATION>
inline std::optional<beliefprop::BpRunOutput> RunBpOnStereoSetSingleThreadCPU<T, DISP_VALS, ACCELERATION>::operator()(const std::array<std::string, 2>& ref_test_image_path,
    const beliefprop::BpSettings& alg_settings, const ParallelParams& parallel_params) const
{
  //return no value if acceleration setting is not NONE
  if constexpr (ACCELERATION != run_environment::AccSetting::kNone) {
    return {};
  }

  //load input
  bp_single_thread_imp::image<uchar> *img1, *img2;
  img1 = loadPGMOrPPMImage(ref_test_image_path[0].c_str());
  img2 = loadPGMOrPPMImage(ref_test_image_path[1].c_str());

  //run single-thread belief propagation implementation and return output
  //disparity map and run data
  std::chrono::duration<double> runtime;
  const auto [output_disp_map, output_run_data] = stereo_ms(img1, img2, alg_settings, runtime);

  //setup run output to return
  std::optional<beliefprop::BpRunOutput> output{beliefprop::BpRunOutput{}};
  output->run_time = runtime;
  output->run_data = output_run_data;
  output->out_disparity_map =
    DisparityMap<float>(
      std::array<unsigned int, 2>{(unsigned int)img1->width(), (unsigned int)img1->height()});

  //set disparity at each point in disparity map from single-thread run output
  for (unsigned int y = 0; y < (unsigned int)img1->height(); y++) {
    for (unsigned int x = 0; x < (unsigned int)img1->width(); x++) {
      output->out_disparity_map.SetPixelAtPoint({x, y}, (float)imRef(output_disp_map, x, y));
    }
  }

  //free dynamically allocated memory
  delete img1;
  delete img2;
  delete output_disp_map;

  //return run output
  return output;
}

#endif /* STEREO_H_ */
