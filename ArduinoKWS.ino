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
#include <TensorFlowLite.h>

#include "audio_provider.h"
#include "command_responder.h"
#include "feature_provider.h"
#include "main_functions.h"
#include "micro_features_micro_model_settings.h"
#include "micro_features_model.h"
#include "recognize_commands.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

#undef PROFILE_MICRO_SPEECH


// Globals, used for compatibility with Arduino-style sketches.
namespace {
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* model_input = nullptr;
FeatureProvider* feature_provider = nullptr;
RecognizeCommands* recognizer = nullptr;
int32_t previous_time = 0;

// Create an area of memory to use for input, output, and intermediate arrays.
// The size of this will depend on the model you're using, and may need to be
// determined by experimentation.
constexpr int kTensorArenaSize = 170 * 1024;
uint8_t tensor_arena[kTensorArenaSize];
int8_t feature_buffer[kFeatureElementCount];
int8_t* model_input_buffer = nullptr;
}  // namespace

// The name of this function is important for Arduino compatibility.
void setup() {
  tflite::InitializeTarget();

  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  model = tflite::GetModel(g_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    MicroPrintf(
        "Model provided is schema version %d not equal "
        "to supported version %d.",
        model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  // Pull in only the operation implementations we need.
  // This relies on a complete list of all the ops needed by this graph.
  // An easier approach is to just use the AllOpsResolver, but this will
  // incur some penalty in code space for op implementations that are not
  // needed by this graph.
  //
  // tflite::AllOpsResolver resolver;
  // NOLINTNEXTLINE(runtime-global-variables)
  // Старые функции
  // static tflite::MicroMutableOpResolver<4> micro_op_resolver;
  // if (micro_op_resolver.AddDepthwiseConv2D() != kTfLiteOk) {
  //   return;
  // }
  // if (micro_op_resolver.AddFullyConnected() != kTfLiteOk) {
  //   return;
  // }
  // if (micro_op_resolver.AddSoftmax() != kTfLiteOk) {
  //   return;
  // }
  // if (micro_op_resolver.AddReshape() != kTfLiteOk) {
  //   return;
  // }
   static tflite::MicroMutableOpResolver<6> micro_op_resolver;
  if (micro_op_resolver.AddDepthwiseConv2D() != kTfLiteOk) {
    return;
  }
  if (micro_op_resolver.AddFullyConnected() != kTfLiteOk) {
    return;
  }
  if (micro_op_resolver.AddSoftmax() != kTfLiteOk) {
    return;
  }
  if (micro_op_resolver.AddReshape() != kTfLiteOk) {
    return;
  }
  if (micro_op_resolver.AddConv2D() != kTfLiteOk) {
    return;
  }
  if (micro_op_resolver.AddAveragePool2D() != kTfLiteOk) {
    return;
  }

  // Build an interpreter to run the model with.
  static tflite::MicroInterpreter static_interpreter(
      model, 
      micro_op_resolver, 
      // resolver,
      tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors.
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    MicroPrintf("AllocateTensors() failed");
    return;
  }

  // Get information about the memory area to use for the model's input.
  model_input = interpreter->input(0);
  if (
    (model_input->dims->size != 4) 
    || (model_input->dims->data[0] != 1) 
    // || (model_input->dims->data[2] != kFeatureSliceCount) 
    // || (model_input->dims->data[1] !=  kFeatureSliceSize) 
    || (model_input->dims->data[1] != kFeatureSliceCount) 
    || (model_input->dims->data[2] !=  kFeatureSliceSize) 
    || (model_input->dims->data[3] != 1) 
    || (model_input->type != kTfLiteInt8)
      ) {
    MicroPrintf("Bad input tensor parameters in model");
    return;
  }
  // Старые условия
  // if (
  //   (model_input->dims->size != 2) 
  //   || (model_input->dims->data[0] != 1) 
  //   || (model_input->dims->data[1] != (kFeatureSliceCount * kFeatureSliceSize)) 
  //   || (model_input->type != kTfLiteInt8)
  //     ) {
  //   MicroPrintf("Bad input tensor parameters in model");
  //   // return;
  // }
  model_input_buffer = model_input->data.int8;

  // Prepare to access the audio spectrograms from a microphone or other source
  // that will provide the inputs to the neural network.
  // NOLINTNEXTLINE(runtime-global-variables)
  static FeatureProvider static_feature_provider(kFeatureElementCount,
                                                 feature_buffer);
  feature_provider = &static_feature_provider;

  static RecognizeCommands static_recognizer;
  recognizer = &static_recognizer;

  previous_time = 0;

  // start the audio
  TfLiteStatus init_status = InitAudioRecording();
  if (init_status != kTfLiteOk) {
    MicroPrintf("Unable to initialize audio");
    return;
  }

  MicroPrintf("Initialization complete");
}

// The name of this function is important for Arduino compatibility.
void loop() {
  // MicroPrintf("NEW");
  // return;
#ifdef PROFILE_MICRO_SPEECH
  const uint32_t prof_start = millis();
  static uint32_t prof_count = 0;
  static uint32_t prof_sum = 0;
  static uint32_t prof_min = std::numeric_limits<uint32_t>::max();
  static uint32_t prof_max = 0;
#endif  // PROFILE_MICRO_SPEECH

  // Fetch the spectrogram for the current time.
  const int32_t current_time = LatestAudioTimestamp();
  int how_many_new_slices = 0;
  // Ошибка происходит тут
  // MicroPrintf("FeatureProvider feature_size  %d", feature_provider->feature_size_);
  TfLiteStatus feature_status = feature_provider->PopulateFeatureData(
      previous_time, current_time, &how_many_new_slices);
  if (feature_status != kTfLiteOk) {
    MicroPrintf("Feature generation failed");
    return;
  }
  previous_time += how_many_new_slices * kFeatureSliceStrideMs;
  // If no new audio samples have been received since last time, don't bother
  // running the network model.
  if (how_many_new_slices == 0) {
    return;
  }

  // Copy feature buffer to input tensor
  // int8_t my_input_0[] = {2, 85, 81, 81, 79, 81, 82, 81, 83, 84, 0, 82, 81, 82, 79, 81, 81, 81, 83, 84, 0, 85, 83, 84, 79, 81, 81, 80, 82, 81, 0, 85, 83, 82, 82, 81, 80, 81, 82, 79, 0, 83, 83, 85, 81, 83, 81, 80, 80, 82, 1, 83, 80, 83, 81, 81, 82, 81, 82, 83, 1, 81, 81, 82, 80, 82, 81, 82, 83, 83, 23, 72, 83, 81, 74, 80, 83, 82, 85, 81, 34, 65, 83, 81, 78, 82, 81, 83, 86, 83, 43, 67, 87, 83, 76, 79, 78, 80, 83, 84, 39, 75, 89, 79, 76, 77, 78, 81, 83, 84, 36, 75, 92, 76, 75, 77, 78, 81, 84, 85, 33, 73, 93, 75, 76, 78, 76, 80, 84, 86, 36, 76, 91, 76, 77, 78, 77, 80, 84, 85, 83, 71, 84, 80, 81, 83, 81, 84, 87, 85, 113, 87, 81, 79, 79, 80, 80, 82, 83, 80, 112, 85, 76, 76, 80, 81, 79, 84, 84, 79, 111, 85, 76, 76, 81, 82, 80, 84, 84, 80, 110, 85, 74, 77, 81, 83, 79, 82, 84, 80, 108, 87, 74, 77, 80, 84, 79, 82, 82, 81, 106, 88, 74, 79, 79, 84, 79, 82, 81, 82, 104, 87, 76, 80, 79, 85, 79, 82, 81, 82, 95, 89, 77, 80, 79, 84, 80, 81, 82, 82, 79, 90, 81, 82, 79, 84, 79, 82, 82, 80, 67, 87, 81, 79, 77, 83, 80, 83, 81, 80, 57, 89, 80, 77, 75, 82, 80, 83, 82, 82, 50, 91, 78, 78, 77, 81, 79, 82, 82, 82, 39, 89, 76, 77, 78, 83, 79, 82, 83, 80, 47, 81, 78, 77, 76, 84, 77, 82, 84, 81, 56, 72, 80, 82, 81, 86, 79, 83, 83, 81, 63, 71, 81, 82, 80, 85, 81, 83, 81, 81, 67, 70, 77, 81, 78, 83, 81, 83, 84, 82, 68, 71, 76, 84, 79, 84, 82, 82, 81, 80, 65, 75, 79, 84, 80, 83, 80, 81, 82, 82, 50, 76, 75, 82, 79, 82, 79, 80, 82, 81, 45, 78, 76, 81, 78, 84, 79, 83, 83, 80, 43, 79, 74, 78, 78, 86, 80, 84, 81, 79, 40, 80, 74, 80, 77, 85, 80, 83, 82, 80, 35, 82, 75, 79, 80, 85, 80, 81, 82, 81, 35, 83, 73, 78, 79, 87, 80, 83, 82, 80, 34, 84, 72, 78, 78, 85, 79, 82, 82, 79, 33, 84, 70, 77, 77, 84, 80, 84, 83, 79, 33, 87, 73, 77, 75, 79, 79, 84, 81, 79, 36, 89, 75, 81, 75, 78, 79, 82, 82, 81, 34, 91, 75, 79, 74, 78, 79, 82, 84, 82, 29, 88, 76, 78, 75, 81, 81, 81, 82, 81, 25, 88, 77, 80, 76, 80, 80, 79, 80, 83, 25, 92, 78, 79, 75, 78, 79, 79, 82, 82, 23, 94, 81, 77, 75, 76, 80, 81, 81, 82};
  // int8_t my_input_1[] = {-8, 82, 83, 82, 84, 85, 83, 82, 84, 83, -6, 81, 86, 84, 84, 82, 82, 83, 83, 83, -8, 81, 84, 83, 83, 83, 83, 80, 82, 83, -7, 79, 83, 81, 82, 80, 81, 81, 83, 82, -8, 80, 85, 83, 85, 82, 82, 82, 82, 82, -8, 79, 84, 83, 85, 83, 84, 84, 83, 82, -10, 86, 85, 82, 82, 83, 82, 82, 83, 82, -10, 83, 87, 83, 83, 84, 82, 81, 82, 83, -9, 83, 84, 83, 83, 83, 84, 82, 82, 82, -13, 84, 85, 83, 84, 83, 84, 83, 83, 83, -11, 82, 85, 83, 85, 84, 84, 82, 83, 82, -8, 83, 87, 82, 82, 80, 84, 82, 85, 81, -8, 82, 87, 83, 84, 81, 84, 81, 83, 83, -8, 83, 85, 83, 83, 82, 84, 81, 81, 83, -8, 81, 86, 84, 83, 84, 83, 83, 83, 83, -10, 84, 85, 83, 82, 82, 83, 82, 82, 81, -8, 81, 86, 84, 83, 83, 83, 82, 83, 82, -6, 81, 84, 82, 81, 81, 83, 82, 81, 84, -6, 82, 84, 82, 82, 81, 81, 83, 82, 84, -5, 82, 85, 82, 82, 81, 83, 84, 83, 83, 5, 95, 94, 86, 84, 82, 82, 83, 81, 82, 52, 105, 95, 81, 81, 79, 83, 81, 82, 82, 66, 103, 90, 81, 83, 77, 83, 80, 85, 82, 77, 98, 92, 81, 83, 77, 83, 80, 85, 82, 85, 96, 92, 79, 84, 77, 82, 81, 85, 82, 96, 93, 90, 77, 83, 79, 82, 82, 85, 81, 104, 92, 88, 77, 83, 79, 84, 82, 86, 79, 107, 91, 86, 78, 82, 79, 84, 82, 86, 77, 109, 90, 86, 79, 82, 79, 84, 84, 85, 78, 110, 90, 85, 80, 82, 79, 84, 84, 85, 77, 111, 89, 85, 81, 81, 79, 85, 84, 85, 77, 109, 89, 86, 80, 81, 80, 85, 84, 85, 77, 107, 90, 85, 82, 81, 79, 85, 84, 84, 78, 103, 89, 85, 83, 81, 81, 86, 83, 84, 78, 96, 92, 84, 83, 82, 82, 86, 84, 83, 79, 86, 92, 86, 85, 82, 81, 83, 84, 83, 79, 71, 91, 85, 87, 85, 81, 84, 84, 83, 80, 63, 86, 84, 84, 81, 80, 85, 82, 84, 80, 53, 83, 84, 87, 80, 81, 88, 82, 83, 81, 52, 85, 87, 86, 82, 81, 85, 83, 83, 80, 41, 81, 90, 89, 82, 83, 83, 81, 83, 81, 40, 83, 89, 89, 81, 82, 85, 84, 82, 81, 32, 82, 88, 88, 82, 80, 85, 81, 83, 83, 21, 77, 89, 85, 83, 79, 83, 81, 82, 82, 8, 80, 88, 86, 81, 79, 82, 82, 84, 83, 3, 81, 86, 83, 81, 80, 83, 82, 83, 83, -3, 82, 83, 82, 81, 81, 82, 83, 82, 81, -4, 82, 83, 81, 83, 82, 84, 84, 83, 83, -5, 81, 85, 81, 83, 83, 83, 82, 82, 81};
  // int8_t my_input_2[] = {31, 81, 83, 80, 80, 81, 82, 81, 82, 81, 37, 82, 81, 81, 80, 82, 83, 82, 84, 84, 47, 84, 82, 80, 79, 81, 83, 81, 83, 83, 54, 83, 82, 79, 78, 81, 83, 80, 83, 83, 49, 82, 84, 80, 78, 81, 84, 81, 82, 84, 48, 79, 82, 79, 78, 78, 82, 81, 80, 84, 48, 77, 82, 82, 80, 81, 84, 81, 81, 84, 49, 76, 81, 83, 81, 81, 84, 81, 80, 83, 49, 79, 81, 83, 81, 80, 84, 82, 79, 83, 45, 79, 80, 84, 82, 81, 84, 81, 80, 83, 39, 82, 80, 83, 82, 81, 81, 82, 81, 81, 35, 85, 80, 83, 84, 84, 83, 84, 82, 81, 31, 84, 80, 81, 83, 82, 82, 81, 82, 81, 29, 85, 83, 81, 81, 82, 81, 82, 82, 82, 26, 85, 84, 83, 82, 82, 82, 82, 83, 83, 26, 83, 83, 84, 83, 82, 81, 81, 81, 81, 26, 84, 83, 84, 83, 81, 80, 82, 83, 82, 49, 98, 88, 83, 81, 79, 79, 82, 83, 83, 72, 106, 81, 79, 84, 79, 80, 82, 82, 83, 81, 105, 78, 79, 85, 77, 81, 82, 82, 83, 90, 100, 77, 81, 82, 79, 81, 82, 84, 84, 95, 97, 75, 82, 80, 82, 82, 83, 85, 80, 97, 95, 75, 80, 80, 83, 83, 84, 84, 79, 96, 93, 75, 81, 80, 83, 85, 83, 81, 79, 95, 93, 75, 81, 79, 84, 86, 83, 81, 79, 89, 94, 77, 80, 79, 84, 85, 84, 81, 79, 86, 94, 80, 79, 77, 83, 84, 84, 83, 81, 83, 94, 84, 79, 77, 82, 82, 83, 84, 81, 82, 93, 87, 81, 78, 81, 80, 83, 83, 81, 79, 92, 90, 82, 77, 81, 79, 83, 83, 81, 72, 95, 89, 83, 78, 81, 79, 84, 81, 80, 61, 97, 92, 84, 79, 82, 81, 83, 83, 80, 56, 99, 90, 85, 80, 81, 81, 84, 81, 79, 45, 93, 87, 81, 80, 80, 81, 83, 81, 79, 40, 89, 85, 81, 80, 79, 81, 83, 83, 80, 36, 88, 85, 80, 79, 79, 81, 81, 81, 79, 58, 86, 80, 77, 80, 83, 84, 82, 83, 80, 70, 83, 84, 79, 80, 83, 83, 84, 84, 82, 71, 82, 82, 81, 79, 82, 83, 83, 83, 84, 67, 82, 82, 79, 77, 80, 84, 82, 81, 83, 60, 83, 80, 81, 78, 82, 85, 82, 82, 81, 54, 83, 80, 82, 79, 81, 84, 83, 82, 81, 46, 84, 81, 82, 82, 80, 82, 81, 79, 82, 40, 85, 80, 83, 82, 81, 81, 82, 80, 83, 36, 86, 81, 81, 83, 81, 82, 81, 81, 83, 31, 85, 81, 81, 83, 81, 84, 84, 83, 83, 29, 84, 83, 80, 81, 80, 82, 82, 84, 83, 30, 84, 84, 83, 82, 81, 82, 82, 83, 82, 29, 83, 84, 83, 83, 81, 81, 81, 83, 82};
  // int8_t my_input_3[] = {-6, 84, 85, 83, 83, 83, 81, 82, 80, 81, -11, 82, 87, 86, 85, 83, 82, 82, 80, 80, -8, 84, 86, 86, 85, 83, 82, 82, 82, 81, -9, 83, 87, 85, 84, 82, 80, 81, 81, 80, -9, 84, 87, 85, 84, 81, 81, 82, 81, 80, -6, 85, 87, 85, 84, 83, 82, 81, 82, 82, -7, 84, 85, 84, 83, 82, 80, 81, 80, 81, -5, 83, 85, 83, 83, 81, 82, 83, 82, 81, -8, 84, 87, 86, 85, 83, 81, 81, 82, 81, -8, 84, 87, 86, 83, 82, 82, 82, 80, 81, -6, 85, 87, 85, 84, 83, 81, 80, 81, 81, -8, 83, 86, 86, 85, 83, 82, 81, 81, 82, -11, 82, 86, 85, 86, 83, 81, 81, 83, 82, -8, 83, 88, 86, 84, 83, 82, 82, 82, 82, -10, 83, 86, 84, 83, 82, 81, 81, 81, 81, -8, 84, 88, 85, 85, 83, 81, 80, 81, 81, -8, 83, 87, 85, 83, 81, 81, 81, 81, 82, -8, 85, 87, 84, 85, 82, 81, 81, 81, 81, -10, 85, 87, 84, 85, 81, 81, 82, 81, 82, -10, 83, 87, 85, 83, 82, 81, 82, 82, 80, -9, 85, 86, 84, 83, 82, 80, 80, 81, 81, -9, 84, 87, 85, 84, 81, 80, 81, 82, 82, -9, 83, 88, 85, 84, 81, 81, 81, 81, 81, -6, 84, 88, 87, 83, 80, 81, 81, 80, 81, -7, 85, 89, 85, 85, 82, 80, 81, 81, 82, -6, 84, 87, 86, 85, 81, 79, 81, 82, 82, 24, 83, 74, 78, 84, 89, 79, 87, 83, 79, 74, 85, 75, 76, 79, 83, 77, 87, 80, 80, 110, 82, 76, 78, 81, 82, 77, 85, 80, 79, 111, 81, 77, 78, 82, 82, 78, 85, 80, 78, 111, 83, 77, 80, 81, 83, 79, 85, 80, 78, 108, 84, 77, 80, 83, 82, 77, 83, 80, 79, 98, 83, 76, 78, 83, 82, 78, 83, 79, 77, 90, 83, 76, 80, 83, 83, 78, 84, 80, 78, 83, 80, 75, 79, 82, 84, 81, 84, 80, 76, 76, 80, 74, 81, 83, 84, 80, 86, 82, 78, 71, 80, 73, 82, 82, 84, 79, 84, 81, 79, 65, 82, 72, 80, 84, 84, 79, 84, 82, 77, 56, 82, 75, 81, 86, 85, 79, 85, 82, 77, 60, 83, 75, 80, 83, 85, 80, 86, 81, 80, 53, 84, 73, 82, 86, 85, 81, 84, 81, 78, 50, 85, 77, 81, 87, 85, 82, 86, 80, 78, 46, 86, 75, 83, 86, 82, 78, 84, 82, 79, 40, 86, 77, 84, 88, 81, 80, 84, 82, 79, 33, 86, 81, 84, 87, 82, 81, 83, 81, 79, 27, 87, 83, 85, 87, 81, 78, 80, 81, 81, 27, 88, 84, 83, 85, 82, 80, 82, 82, 82, 23, 85, 84, 86, 84, 81, 80, 81, 81, 80, 23, 85, 85, 86, 84, 80, 79, 81, 82, 82};
  // int8_t my_input_4[] = {0, 81, 90, 81, 83, 80, 75, 83, 77, 83, 10, 77, 91, 82, 83, 73, 77, 86, 76, 84, 14, 80, 88, 81, 78, 73, 76, 85, 80, 85, 14, 79, 87, 80, 78, 74, 78, 84, 81, 86, 1, 80, 86, 82, 80, 79, 80, 82, 82, 85, -5, 82, 84, 81, 82, 82, 80, 83, 84, 82, -10, 82, 85, 84, 83, 81, 82, 82, 82, 82, -9, 82, 82, 82, 83, 82, 81, 83, 83, 83, -8, 83, 82, 80, 81, 81, 81, 81, 82, 82, -11, 83, 85, 82, 83, 84, 81, 81, 81, 82, -10, 83, 83, 82, 82, 79, 81, 81, 81, 81, -10, 84, 83, 82, 82, 81, 81, 81, 83, 82, -11, 84, 83, 83, 82, 81, 81, 83, 83, 83, -8, 84, 84, 82, 81, 79, 81, 84, 85, 83, -11, 84, 84, 82, 83, 81, 80, 82, 84, 84, -5, 87, 80, 77, 82, 82, 83, 82, 80, 81, 26, 101, 75, 70, 80, 87, 83, 80, 79, 80, 46, 107, 76, 72, 84, 81, 81, 79, 80, 79, 79, 98, 78, 72, 84, 83, 82, 76, 84, 78, 96, 93, 77, 74, 83, 85, 83, 76, 83, 79, 97, 92, 79, 75, 83, 83, 84, 75, 81, 78, 101, 93, 78, 75, 83, 83, 83, 75, 81, 77, 102, 93, 77, 75, 81, 83, 82, 75, 82, 77, 102, 91, 78, 76, 83, 83, 82, 74, 81, 77, 102, 92, 78, 76, 83, 84, 82, 75, 80, 78, 101, 92, 80, 77, 81, 85, 82, 77, 80, 78, 101, 92, 80, 78, 80, 85, 82, 78, 80, 77, 101, 94, 79, 79, 79, 85, 80, 79, 82, 78, 91, 97, 81, 80, 81, 87, 79, 80, 82, 78, 84, 99, 83, 83, 81, 86, 79, 80, 82, 78, 79, 101, 83, 84, 79, 88, 78, 80, 83, 79, 70, 105, 84, 85, 79, 88, 78, 79, 83, 78, 60, 106, 86, 84, 81, 91, 77, 81, 83, 77, 50, 105, 90, 86, 78, 90, 79, 81, 82, 79, 43, 105, 90, 86, 81, 89, 81, 81, 82, 79, 35, 100, 91, 90, 82, 86, 82, 82, 81, 81, 27, 97, 92, 89, 83, 86, 82, 83, 80, 80, 18, 95, 91, 87, 84, 86, 83, 83, 81, 83, 12, 92, 92, 89, 85, 85, 84, 83, 80, 81, 8, 92, 91, 87, 87, 84, 83, 81, 81, 82, 0, 88, 86, 83, 84, 85, 83, 82, 81, 81, -3, 86, 84, 82, 85, 84, 82, 81, 79, 81, -6, 85, 86, 83, 82, 83, 83, 81, 81, 82, -8, 84, 84, 84, 83, 80, 82, 83, 83, 82, -6, 84, 83, 83, 83, 83, 83, 85, 83, 81, -10, 82, 84, 82, 81, 82, 81, 82, 83, 82, -10, 84, 86, 84, 82, 80, 80, 83, 83, 83, -10, 84, 85, 84, 81, 81, 82, 81, 82, 82, -8, 85, 85, 84, 82, 80, 80, 80, 82, 83};
  // int8_t my_input_5[] = {-4, 83, 81, 88, 84, 78, 83, 81, 78, 82, -2, 83, 80, 88, 83, 76, 81, 80, 75, 81, 1, 84, 80, 88, 83, 73, 81, 80, 76, 80, 1, 84, 78, 88, 83, 75, 82, 81, 76, 82, -2, 81, 77, 89, 83, 76, 84, 84, 79, 83, 0, 80, 75, 88, 83, 78, 85, 81, 78, 82, 0, 80, 79, 92, 85, 79, 88, 81, 78, 81, 3, 83, 78, 90, 84, 80, 89, 82, 79, 82, 2, 83, 78, 90, 85, 79, 86, 81, 78, 83, 1, 82, 78, 88, 85, 79, 86, 83, 80, 84, -8, 82, 81, 87, 85, 81, 87, 82, 79, 84, -10, 79, 81, 83, 83, 81, 85, 84, 79, 84, -3, 73, 84, 84, 83, 80, 83, 84, 78, 83, -6, 78, 85, 82, 82, 81, 83, 83, 80, 82, -11, 82, 84, 82, 83, 82, 82, 83, 83, 82, -20, 81, 86, 83, 82, 82, 83, 83, 82, 81, -13, 83, 85, 81, 83, 83, 83, 82, 81, 81, -14, 83, 86, 82, 82, 83, 81, 80, 81, 80, -14, 84, 86, 83, 83, 82, 81, 79, 81, 81, -16, 84, 85, 85, 84, 81, 82, 81, 81, 82, -14, 83, 84, 82, 83, 82, 82, 81, 82, 82, -13, 83, 85, 83, 84, 82, 82, 82, 83, 83, 12, 85, 80, 80, 80, 84, 83, 82, 82, 83, 85, 88, 82, 82, 81, 81, 85, 79, 79, 81, 110, 86, 87, 80, 79, 78, 84, 82, 81, 83, 113, 86, 84, 77, 81, 81, 86, 82, 80, 81, 117, 85, 83, 78, 81, 82, 85, 82, 79, 80, 112, 84, 81, 77, 82, 83, 84, 82, 77, 80, 113, 84, 80, 77, 83, 82, 82, 82, 78, 80, 112, 85, 81, 78, 84, 82, 81, 82, 78, 81, 113, 86, 81, 76, 84, 81, 82, 82, 79, 81, 111, 86, 80, 76, 84, 81, 81, 82, 79, 81, 113, 87, 79, 76, 85, 81, 81, 82, 78, 81, 113, 87, 79, 76, 85, 81, 81, 82, 79, 81, 109, 88, 80, 76, 85, 81, 81, 81, 80, 81, 110, 88, 79, 76, 84, 80, 80, 80, 80, 80, 108, 91, 79, 76, 84, 80, 81, 80, 81, 81, 107, 91, 81, 76, 83, 81, 82, 80, 80, 82, 101, 92, 82, 76, 82, 82, 82, 79, 81, 83, 98, 93, 81, 77, 82, 81, 83, 80, 80, 82, 97, 93, 83, 76, 82, 81, 83, 80, 80, 82, 94, 94, 82, 77, 81, 81, 82, 81, 80, 82, 93, 93, 82, 78, 82, 81, 82, 81, 81, 82, 93, 93, 83, 78, 82, 82, 81, 81, 80, 84, 91, 93, 84, 79, 81, 81, 80, 81, 81, 83, 86, 92, 85, 82, 80, 81, 80, 82, 81, 83, 86, 92, 86, 82, 80, 79, 81, 81, 82, 81, 86, 92, 86, 82, 80, 80, 79, 80, 82, 82, 84, 92, 87, 82, 80, 80, 79, 81, 81, 82};
  // int8_t my_input_6[] = {37, 79, 82, 84, 82, 81, 83, 83, 82, 80, 35, 79, 83, 84, 81, 81, 83, 82, 82, 81, 34, 80, 84, 82, 83, 83, 84, 83, 83, 81, 33, 81, 84, 81, 83, 83, 82, 82, 83, 83, 38, 82, 84, 82, 83, 84, 84, 80, 81, 83, 34, 81, 85, 82, 83, 82, 83, 80, 82, 83, 39, 81, 84, 85, 82, 80, 82, 80, 82, 82, 48, 87, 82, 83, 82, 78, 79, 79, 83, 82, 43, 83, 82, 83, 82, 79, 79, 81, 82, 80, 36, 82, 85, 83, 84, 84, 82, 82, 82, 81, 41, 84, 87, 84, 85, 83, 81, 79, 79, 81, 62, 85, 78, 82, 84, 81, 81, 82, 81, 81, 119, 84, 75, 81, 85, 81, 80, 81, 81, 80, 113, 87, 74, 81, 85, 81, 80, 81, 81, 81, 111, 90, 72, 82, 83, 82, 81, 81, 81, 80, 109, 91, 72, 82, 83, 83, 81, 81, 82, 80, 108, 92, 73, 80, 84, 82, 82, 81, 82, 79, 108, 93, 74, 79, 83, 81, 82, 81, 83, 78, 92, 96, 78, 79, 81, 82, 81, 83, 85, 80, 67, 87, 81, 82, 84, 84, 80, 82, 81, 81, 54, 84, 79, 81, 84, 85, 82, 83, 81, 80, 51, 82, 80, 80, 82, 84, 81, 81, 82, 82, 49, 81, 81, 81, 82, 85, 84, 83, 81, 81, 43, 81, 81, 82, 82, 82, 81, 81, 79, 80, 44, 80, 82, 82, 84, 83, 83, 83, 82, 82, 44, 81, 85, 84, 83, 82, 82, 85, 82, 80, 59, 87, 85, 87, 83, 82, 84, 85, 84, 83, 60, 89, 82, 84, 83, 83, 86, 83, 82, 83, 55, 84, 79, 82, 83, 82, 83, 83, 81, 83, 49, 84, 79, 80, 84, 82, 82, 81, 81, 81, 48, 82, 79, 80, 83, 84, 83, 80, 80, 82, 45, 82, 81, 81, 83, 83, 82, 81, 83, 82, 42, 81, 82, 82, 85, 85, 81, 82, 81, 83, 42, 82, 85, 84, 84, 83, 81, 81, 83, 83, 39, 82, 82, 83, 83, 82, 84, 83, 82, 84, 40, 81, 83, 85, 83, 83, 83, 82, 81, 83, 42, 80, 85, 84, 83, 84, 82, 81, 81, 82, 42, 80, 84, 82, 82, 81, 81, 82, 82, 83, 40, 79, 82, 82, 81, 81, 81, 82, 83, 85, 41, 81, 82, 83, 82, 83, 82, 81, 84, 84, 39, 78, 81, 81, 85, 84, 81, 82, 80, 82, 38, 79, 83, 83, 85, 84, 82, 83, 80, 82, 41, 81, 83, 83, 83, 83, 83, 82, 82, 83, 40, 79, 83, 84, 83, 82, 81, 82, 83, 84, 43, 81, 84, 85, 84, 83, 81, 83, 83, 82, 40, 79, 84, 84, 83, 84, 81, 81, 82, 83, 37, 80, 83, 84, 82, 83, 81, 81, 81, 83, 37, 79, 82, 85, 85, 84, 84, 82, 81, 82, 35, 80, 81, 83, 83, 83, 83, 81, 82, 82};
  // int8_t my_input_7[] = {21, 75, 76, 80, 80, 85, 83, 81, 80, 82, 24, 76, 76, 79, 76, 85, 85, 84, 82, 84, 29, 77, 76, 80, 79, 85, 83, 80, 81, 83, 32, 81, 76, 79, 82, 82, 81, 80, 81, 80, 24, 79, 81, 79, 83, 86, 82, 79, 78, 81, 32, 81, 81, 83, 85, 85, 81, 77, 78, 83, 65, 96, 97, 83, 83, 82, 79, 80, 79, 84, 85, 90, 99, 83, 77, 80, 76, 81, 79, 84, 96, 81, 96, 83, 74, 80, 76, 78, 77, 83, 100, 81, 93, 82, 73, 80, 78, 82, 79, 85, 104, 82, 92, 82, 72, 81, 79, 81, 79, 85, 108, 84, 90, 82, 73, 82, 81, 82, 80, 84, 105, 85, 87, 81, 75, 81, 82, 81, 80, 83, 94, 86, 87, 82, 76, 80, 82, 83, 81, 83, 82, 89, 86, 83, 77, 80, 80, 82, 81, 82, 80, 85, 90, 84, 79, 79, 85, 81, 83, 82, 77, 81, 92, 84, 82, 79, 83, 81, 79, 82, 73, 81, 90, 85, 84, 82, 86, 83, 79, 83, 70, 83, 90, 81, 84, 79, 85, 83, 78, 83, 68, 80, 90, 80, 84, 79, 84, 82, 80, 84, 69, 78, 91, 77, 86, 82, 81, 82, 81, 82, 62, 71, 90, 82, 88, 86, 83, 80, 80, 84, 62, 70, 87, 79, 84, 81, 82, 80, 75, 83, 64, 73, 88, 81, 87, 83, 83, 82, 80, 83, 61, 75, 89, 78, 86, 84, 81, 83, 80, 82, 53, 68, 85, 74, 84, 83, 82, 83, 80, 83, 48, 67, 85, 75, 84, 86, 85, 81, 81, 83, 50, 63, 81, 76, 84, 85, 83, 80, 79, 83, 75, 75, 85, 77, 82, 84, 85, 78, 78, 81, 67, 74, 85, 75, 80, 82, 84, 79, 80, 82, 50, 68, 82, 77, 84, 82, 83, 80, 79, 82, 34, 70, 80, 79, 84, 85, 84, 79, 82, 82, 28, 75, 80, 81, 86, 87, 84, 80, 85, 82, 27, 75, 78, 80, 82, 84, 82, 79, 80, 84, 27, 79, 78, 81, 84, 84, 82, 79, 78, 81, 27, 78, 77, 84, 82, 86, 83, 82, 78, 84, 27, 77, 78, 81, 83, 84, 83, 81, 80, 86, 26, 75, 76, 80, 82, 88, 84, 81, 82, 84, 29, 78, 81, 84, 86, 88, 82, 79, 81, 84, 28, 79, 81, 86, 86, 86, 80, 79, 79, 84, 29, 78, 77, 82, 87, 87, 81, 81, 81, 82, 24, 78, 78, 80, 84, 85, 82, 84, 83, 85, 25, 73, 74, 79, 83, 87, 83, 80, 82, 84, 22, 71, 77, 79, 82, 87, 85, 78, 81, 84, 23, 77, 75, 78, 80, 80, 80, 81, 82, 85, 26, 81, 78, 80, 82, 84, 82, 79, 83, 83, 27, 83, 80, 84, 84, 84, 81, 81, 81, 85, 19, 73, 76, 79, 85, 86, 82, 80, 80, 83, 21, 75, 78, 80, 83, 86, 84, 79, 77, 79};
  // int8_t my_input_8[] = {31, 88, 88, 83, 83, 80, 82, 82, 82, 82, 31, 86, 85, 83, 84, 82, 82, 81, 81, 82, 32, 84, 85, 82, 83, 81, 82, 83, 80, 82, 37, 86, 86, 83, 83, 80, 81, 83, 83, 81, 36, 86, 87, 82, 83, 80, 81, 82, 83, 81, 33, 86, 86, 82, 83, 82, 80, 81, 82, 82, 32, 85, 86, 82, 82, 78, 79, 81, 84, 82, 33, 86, 86, 81, 83, 81, 81, 82, 82, 82, 34, 87, 86, 82, 82, 81, 81, 83, 83, 83, 31, 87, 86, 82, 83, 79, 82, 82, 83, 82, 32, 87, 87, 83, 83, 79, 82, 83, 82, 82, 31, 88, 86, 82, 83, 80, 81, 83, 83, 83, 33, 89, 86, 81, 84, 81, 82, 83, 83, 82, 32, 88, 87, 81, 82, 81, 81, 80, 82, 81, 32, 88, 87, 82, 83, 80, 82, 83, 85, 83, 33, 86, 86, 82, 82, 80, 81, 83, 84, 83, 34, 88, 86, 82, 84, 81, 81, 83, 82, 82, 31, 87, 86, 82, 82, 79, 81, 82, 83, 82, 73, 78, 77, 75, 88, 83, 80, 82, 79, 79, 103, 84, 77, 76, 84, 81, 81, 81, 81, 81, 102, 84, 78, 77, 84, 82, 80, 81, 81, 82, 104, 82, 76, 77, 86, 80, 80, 80, 81, 81, 108, 81, 76, 77, 85, 81, 81, 81, 81, 81, 101, 81, 76, 77, 86, 82, 81, 81, 82, 81, 104, 81, 74, 75, 85, 80, 80, 81, 82, 82, 101, 80, 73, 75, 86, 80, 80, 82, 82, 83, 102, 80, 73, 74, 87, 82, 80, 80, 83, 81, 103, 82, 75, 75, 85, 82, 81, 80, 82, 83, 102, 81, 74, 76, 85, 80, 80, 81, 82, 80, 101, 84, 76, 77, 85, 81, 79, 81, 81, 81, 96, 85, 77, 81, 87, 83, 80, 81, 80, 81, 89, 83, 78, 79, 86, 82, 80, 82, 81, 82, 85, 78, 75, 77, 85, 81, 81, 81, 82, 82, 81, 76, 77, 79, 86, 82, 81, 82, 81, 83, 78, 78, 80, 80, 87, 84, 80, 82, 81, 82, 72, 77, 79, 81, 87, 82, 80, 82, 80, 81, 69, 78, 80, 83, 87, 82, 78, 81, 81, 84, 68, 77, 82, 84, 87, 81, 78, 80, 81, 83, 69, 78, 82, 82, 86, 81, 79, 82, 81, 83, 69, 75, 84, 84, 83, 83, 82, 81, 82, 83, 69, 75, 84, 83, 83, 81, 82, 82, 82, 84, 67, 77, 86, 84, 82, 81, 83, 83, 81, 83, 69, 76, 85, 83, 82, 81, 82, 82, 82, 83, 67, 77, 85, 82, 82, 81, 83, 80, 81, 83, -2, 76, 85, 76, 85, 79, 81, 84, 81, 83, -128, 82, 83, 83, 83, 82, 83, 83, 83, 83, -128, 82, 83, 83, 83, 82, 83, 83, 83, 83, -128, 82, 83, 83, 83, 82, 83, 83, 83, 83, -128, 82, 83, 83, 83, 82, 83, 83, 83, 83};
  // int8_t my_input_9[] = {-4, 86, 84, 82, 82, 81, 81, 80, 81, 80, -4, 86, 87, 82, 81, 82, 82, 80, 80, 80, -5, 86, 84, 82, 82, 81, 79, 79, 82, 80, -4, 85, 86, 85, 82, 82, 79, 81, 82, 80, -5, 87, 87, 83, 81, 80, 81, 81, 81, 80, -1, 89, 85, 82, 82, 81, 80, 80, 80, 80, -5, 86, 85, 83, 81, 81, 80, 80, 81, 80, -3, 88, 86, 84, 83, 81, 80, 81, 81, 80, 0, 89, 86, 84, 82, 82, 80, 80, 81, 81, 56, 101, 89, 90, 71, 80, 86, 75, 84, 81, 97, 90, 82, 84, 81, 77, 84, 78, 83, 78, 103, 91, 80, 82, 81, 76, 85, 79, 83, 78, 106, 91, 79, 83, 81, 78, 83, 81, 82, 78, 109, 91, 78, 81, 80, 78, 82, 80, 82, 79, 109, 91, 78, 80, 81, 78, 82, 82, 83, 80, 109, 89, 78, 79, 82, 80, 82, 81, 83, 80, 107, 89, 79, 79, 82, 80, 82, 80, 83, 81, 105, 90, 79, 79, 82, 81, 82, 80, 84, 82, 100, 92, 79, 79, 82, 82, 83, 80, 83, 82, 94, 93, 80, 79, 81, 81, 82, 79, 82, 82, 91, 94, 81, 80, 81, 84, 83, 80, 82, 80, 84, 92, 79, 82, 82, 82, 84, 80, 84, 81, 77, 93, 79, 82, 83, 82, 84, 80, 85, 80, 72, 91, 77, 83, 82, 82, 83, 80, 83, 80, 69, 90, 75, 82, 83, 83, 84, 79, 85, 80, 66, 91, 77, 85, 83, 83, 84, 81, 85, 79, 63, 91, 79, 85, 83, 83, 83, 78, 85, 78, 58, 90, 76, 84, 84, 84, 81, 79, 84, 79, 52, 92, 79, 82, 83, 82, 84, 80, 83, 79, 51, 93, 77, 81, 83, 84, 81, 81, 84, 80, 46, 89, 78, 81, 83, 83, 81, 81, 85, 82, 41, 87, 76, 81, 81, 81, 80, 81, 84, 78, 38, 88, 79, 82, 83, 80, 79, 81, 84, 79, 34, 87, 80, 81, 82, 82, 82, 81, 83, 80, 30, 88, 81, 81, 81, 82, 81, 80, 84, 81, 25, 88, 81, 82, 82, 83, 82, 80, 83, 80, 25, 89, 83, 82, 82, 82, 80, 81, 85, 81, 23, 89, 84, 80, 79, 83, 81, 79, 82, 79, 22, 89, 83, 81, 79, 83, 80, 80, 82, 80, 20, 90, 84, 83, 80, 83, 81, 80, 80, 81, 15, 85, 81, 78, 78, 79, 78, 80, 84, 83, 14, 86, 82, 83, 81, 80, 78, 82, 83, 81, 15, 86, 84, 83, 81, 80, 78, 82, 84, 83, 15, 88, 82, 83, 81, 82, 79, 82, 81, 81, 14, 88, 84, 84, 81, 81, 79, 79, 82, 80, 15, 89, 84, 84, 81, 80, 80, 80, 83, 80, 15, 91, 88, 87, 81, 81, 80, 80, 84, 81, 9, 89, 87, 85, 82, 81, 82, 79, 80, 79, 5, 84, 83, 81, 80, 81, 82, 80, 80, 81};
  // for (int i = 0; i < kFeatureElementCount; i++) {
  //   model_input_buffer[i] = my_input_9[i];
  // }
  for (int i = 0; i < kFeatureElementCount; i++) {
    model_input_buffer[i] = feature_buffer[i];
  }
  
  // Run the model on the spectrogram input and make sure it succeeds.
  TfLiteStatus invoke_status = interpreter->Invoke();
  if (invoke_status != kTfLiteOk) {
    MicroPrintf("Invoke failed");
    return;
  }

  // Obtain a pointer to the output tensor
  TfLiteTensor* output = interpreter->output(0);
  // Determine whether a command was recognized based on the output of inference
  const char* found_command = nullptr;
  uint8_t score = 0;
  bool is_new_command = false;
  TfLiteStatus process_status = recognizer->ProcessLatestResults(
      output, current_time, &found_command, &score, &is_new_command);
  if (process_status != kTfLiteOk) {
    MicroPrintf("RecognizeCommands::ProcessLatestResults() failed");
    return;
  }
  // Do something based on the recognized command. The default implementation
  // just prints to the error console, but you should replace this with your
  // own function for a real application.
  RespondToCommand(current_time, found_command, score, is_new_command);

#ifdef PROFILE_MICRO_SPEECH
  const uint32_t prof_end = millis();
  if (++prof_count > 10) {
    uint32_t elapsed = prof_end - prof_start;
    prof_sum += elapsed;
    if (elapsed < prof_min) {
      prof_min = elapsed;
    }
    if (elapsed > prof_max) {
      prof_max = elapsed;
    }
    if (prof_count % 300 == 0) {
      MicroPrintf("## time: min %dms  max %dms  avg %dms", prof_min, prof_max,
                  prof_sum / prof_count);
    }
  }
#endif  // PROFILE_MICRO_SPEECH
}
