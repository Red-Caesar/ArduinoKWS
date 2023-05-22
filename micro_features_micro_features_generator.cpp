/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include "micro_features_micro_features_generator.h"

#include <cmath>
#include <cstring>

#include "micro_features_micro_model_settings.h"
// #include "tensorflow/lite/experimental/microfrontend/lib/frontend.h"
#include "tensorflow/lite/experimental/microfrontend/lib/frontend_util.h"
#include "tensorflow/lite/micro/micro_log.h"

// Configure FFT to output 16 bit fixed point.
#define FIXED_POINT 16

namespace {

FrontendState g_micro_features_state;
bool g_is_first_time = true;

}  // namespace

// Зона кода из фронтенда

#include "tensorflow/lite/experimental/microfrontend/lib/frontend.h"
#include "tensorflow/lite/experimental/microfrontend/lib/bits.h"


struct FrontendOutput FrontendProcessSamples(struct FrontendState* state,
                                             const int16_t* samples,
                                             size_t num_samples,
                                             size_t* num_samples_read) {
  struct FrontendOutput output;
  output.values = NULL;
  output.size = 0;

  // Try to apply the window - if it fails, return and wait for more data.
  if (!WindowProcessSamples(&state->window, samples, num_samples,
                            num_samples_read)) {
    return output;
  }

  // Apply the FFT to the window's output (and scale it so that the fixed point
  // FFT can have as much resolution as possible).
  int input_shift =
      15 - MostSignificantBit32(state->window.max_abs_output_value);

  // MicroPrintf("input_shift: %d", input_shift);

  FftCompute(&state->fft, state->window.output, input_shift);

  // MicroPrintf("real %d", state->fft.output[0].real);
  // MicroPrintf("imag %d", state->fft.output[0].imag);

  // for (size_t i = 0; i < state->fft.fft_size; ++i) {
  //   MicroPrintf("%d", state->fft.output[i].real);
  // }
  
  // We can re-ruse the fft's output buffer to hold the energy.
  int32_t* energy = (int32_t*)state->fft.output;

// Конвертируют комплексные числа
// energy: (real * real) + (imag * imag);
  FilterbankConvertFftComplexToEnergy(&state->filterbank, state->fft.output,
                                      energy);

  // for (size_t i = 0; i < state->fft.fft_size; ++i) {
  //   MicroPrintf("%d", energy[i]);
  // }
  

  FilterbankAccumulateChannels(&state->filterbank, energy);
  uint32_t* scaled_filterbank = FilterbankSqrt(&state->filterbank, input_shift);

    // MicroPrintf("scaled_filterbank %d", scaled_filterbank[0]);
    // MicroPrintf("energy %d", energy[0]);
  
  // Apply noise reduction.
  NoiseReductionApply(&state->noise_reduction, scaled_filterbank);

  if (state->pcan_gain_control.enable_pcan) {
    PcanGainControlApply(&state->pcan_gain_control, scaled_filterbank);
  }

  // Apply the log and scale.
  int correction_bits =
      MostSignificantBit32(state->fft.fft_size) - 1 - (kFilterbankBits / 2);
  uint16_t* logged_filterbank =
      LogScaleApply(&state->log_scale, scaled_filterbank,
                    state->filterbank.num_channels, correction_bits);

  output.size = state->filterbank.num_channels;
  output.values = logged_filterbank;

  // for (size_t i = 0; i < output.size; ++i) {
	// MicroPrintf("fronted_output: %d", output.values[i]);
  
  // }
  // MicroPrintf("fronted_output_size: %d", output.size);
  return output;
}

// Конец зоны

TfLiteStatus InitializeMicroFeatures() {
  FrontendConfig config;
  config.window.size_ms = kFeatureSliceDurationMs;
  config.window.step_size_ms = kFeatureSliceStrideMs;
  config.noise_reduction.smoothing_bits = 10;
  config.filterbank.num_channels = kFeatureSliceSize;
  config.filterbank.lower_band_limit = 125.0;
  config.filterbank.upper_band_limit = 7500.0;
  config.noise_reduction.smoothing_bits = 10;
  config.noise_reduction.even_smoothing = 0.025;
  config.noise_reduction.odd_smoothing = 0.06;
  config.noise_reduction.min_signal_remaining = 0.05;
  config.pcan_gain_control.enable_pcan = 1;
  config.pcan_gain_control.strength = 0.95;
  config.pcan_gain_control.offset = 80.0;
  config.pcan_gain_control.gain_bits = 21;
  config.log_scale.enable_log = 1;
  config.log_scale.scale_shift = 6;
  if (!FrontendPopulateState(&config, &g_micro_features_state,
                             kAudioSampleFrequency)) {
    MicroPrintf("FrontendPopulateState() failed");
    return kTfLiteError;
  }
  g_is_first_time = true;
  return kTfLiteOk;
}

// This is not exposed in any header, and is only used for testing, to ensure
// that the state is correctly set up before generating results.
void SetMicroFeaturesNoiseEstimates(const uint32_t* estimate_presets) {
  for (int i = 0; i < g_micro_features_state.filterbank.num_channels; ++i) {
    g_micro_features_state.noise_reduction.estimate[i] = estimate_presets[i];
  }
}

TfLiteStatus GenerateMicroFeatures(const int16_t* input, int input_size,
                                   int output_size, int8_t* output,
                                   size_t* num_samples_read) {
  const int16_t* frontend_input;
  if (g_is_first_time) {
    frontend_input = input;
    g_is_first_time = false;
  } else {
    frontend_input = input;
  }

  // for (size_t i = 0; i < input_size; ++i) {
  //   MicroPrintf("Value: %d", frontend_input[i]);
  // }

  // MicroPrintf("num_samples_read: %d", *num_samples_read);
  FrontendOutput frontend_output = FrontendProcessSamples(
      &g_micro_features_state, frontend_input, input_size, num_samples_read);
  
  // MicroPrintf("END");

  for (size_t i = 0; i < frontend_output.size; ++i) {
    // These scaling values are derived from those used in input_data.py in the
    // training pipeline.
    // The feature pipeline outputs 16-bit signed integers in roughly a 0 to 670
    // range. In training, these are then arbitrarily divided by 25.6 to get
    // float values in the rough range of 0.0 to 26.0. This scaling is performed
    // for historical reasons, to match up with the output of other feature
    // generators.
    // The process is then further complicated when we quantize the model. This
    // means we have to scale the 0.0 to 26.0 real values to the -128 to 127
    // signed integer numbers.
    // All this means that to get matching values from our integer feature
    // output into the tensor input, we have to perform:
    // input = (((feature / 25.6) / 26.0) * 256) - 128
    // To simplify this and perform it in 32-bit integer math, we rearrange to:
    // input = (feature * 256) / (25.6 * 26.0) - 128
    // MicroPrintf("fronted_output: %d", frontend_output.values[i]);
    constexpr int32_t value_scale = 256;
    constexpr int32_t value_div = static_cast<int32_t>((25.6f * 26.0f) + 0.5f);
    int32_t value =
        ((frontend_output.values[i] * value_scale) + (value_div / 2)) /
        value_div;
    value -= 128;
    // float input_scale = 0.5847029089927673;
    // int32_t zero_point = 83;
    // int32_t value = (frontend_output.values[i] / 670)/ input_scale + zero_point;

    if (value < -128) {
      value = -128;
    }
    if (value > 127) {
      value = 127;
    }
    // MicroPrintf("Value: %d,  fronted_output: %d", value, frontend_output.values[i]);
    output[i] = value;
  }

  return kTfLiteOk;
}
