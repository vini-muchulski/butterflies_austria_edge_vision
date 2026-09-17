/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/micro/kernels/fully_connected.h"

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/reference/fully_connected.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/fully_connected.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/micro_log.h"

#if ESP_NN
#include <esp_nn.h>
#endif

#include <esp_timer.h>

#include <algorithm>

long long fc_total_time = 0;

namespace tflite {
namespace {

struct NodeData {
  OpDataFullyConnected op_data;
  int32_t* per_channel_output_multiplier;
  int* per_channel_output_shift;
  int output_depth;
};

uint32_t Fnv1a32(const void* data, size_t size) {
  const auto* bytes = static_cast<const uint8_t*>(data);
  uint32_t hash = 2166136261u;
  for (size_t i = 0; i < size; ++i) {
    hash = (hash ^ bytes[i]) * 16777619u;
  }
  return hash;
}

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  TFLITE_DCHECK(context->AllocatePersistentBuffer != nullptr);
  auto* data = static_cast<NodeData*>(
      context->AllocatePersistentBuffer(context, sizeof(NodeData)));
  if (data == nullptr) {
    return nullptr;
  }
  *data = {};
  return data;
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  MicroContext* micro_context = GetMicroContext(context);

  TFLITE_DCHECK(node->user_data != nullptr);
  TFLITE_DCHECK(node->builtin_data != nullptr);

  auto* data = static_cast<NodeData*>(node->user_data);
  const auto params =
      static_cast<const TfLiteFullyConnectedParams*>(node->builtin_data);

  TfLiteTensor* input =
      micro_context->AllocateTempInputTensor(node, kFullyConnectedInputTensor);
  TF_LITE_ENSURE(context, input != nullptr);
  TfLiteTensor* filter = micro_context->AllocateTempInputTensor(
      node, kFullyConnectedWeightsTensor);
  TF_LITE_ENSURE(context, filter != nullptr);
  TfLiteTensor* bias =
      micro_context->AllocateTempInputTensor(node, kFullyConnectedBiasTensor);
  TfLiteTensor* output = micro_context->AllocateTempOutputTensor(
      node, kFullyConnectedOutputTensor);
  TF_LITE_ENSURE(context, output != nullptr);

  TF_LITE_ENSURE_TYPES_EQ(context, input->type, output->type);
  TF_LITE_ENSURE_MSG(context, input->type == filter->type,
                     "Hybrid models are not supported on TFLite Micro.");

  TF_LITE_ENSURE_OK(context, CalculateOpDataFullyConnected(
                                 context, params->activation, input->type,
                                 input, filter, bias, output, &data->op_data));

  if (input->type == kTfLiteInt8) {
    TF_LITE_ENSURE_EQ(context, filter->quantization.type,
                      kTfLiteAffineQuantization);
    const auto* quantization = static_cast<TfLiteAffineQuantization*>(
        filter->quantization.params);
    TF_LITE_ENSURE(context, quantization != nullptr);
    TF_LITE_ENSURE(context, quantization->scale != nullptr);

    if (quantization->scale->size > 1) {
      data->output_depth = filter->dims->data[filter->dims->size - 2];
      TF_LITE_ENSURE_EQ(context, quantization->scale->size,
                        data->output_depth);
      data->per_channel_output_multiplier =
          static_cast<int32_t*>(context->AllocatePersistentBuffer(
              context, data->output_depth * sizeof(int32_t)));
      data->per_channel_output_shift =
          static_cast<int*>(context->AllocatePersistentBuffer(
              context, data->output_depth * sizeof(int)));
      TF_LITE_ENSURE(context,
                     data->per_channel_output_multiplier != nullptr);
      TF_LITE_ENSURE(context, data->per_channel_output_shift != nullptr);

      for (int i = 0; i < data->output_depth; ++i) {
        const double multiplier =
            static_cast<double>(input->params.scale) *
            static_cast<double>(quantization->scale->data[i]) /
            static_cast<double>(output->params.scale);
        QuantizeMultiplier(multiplier,
                           &data->per_channel_output_multiplier[i],
                           &data->per_channel_output_shift[i]);
      }
    }
  }

  micro_context->DeallocateTempTfLiteTensor(input);
  micro_context->DeallocateTempTfLiteTensor(filter);
  if (bias != nullptr) {
    micro_context->DeallocateTempTfLiteTensor(bias);
  }
  micro_context->DeallocateTempTfLiteTensor(output);
  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  MicroPrintf("FC_DIAG node=%x builtin=%x user=%x inputs=%x outputs=%x",
              static_cast<unsigned>(reinterpret_cast<uintptr_t>(node)),
              static_cast<unsigned>(
                  reinterpret_cast<uintptr_t>(node->builtin_data)),
              static_cast<unsigned>(reinterpret_cast<uintptr_t>(node->user_data)),
              static_cast<unsigned>(reinterpret_cast<uintptr_t>(node->inputs)),
              static_cast<unsigned>(reinterpret_cast<uintptr_t>(node->outputs)));
  if (node->builtin_data == nullptr) {
    MicroPrintf("FC_DIAG builtin_data_null");
    return kTfLiteError;
  }
  const auto* params =
      static_cast<const TfLiteFullyConnectedParams*>(node->builtin_data);
  const auto* node_data_ptr = static_cast<const NodeData*>(node->user_data);
  if (node_data_ptr == nullptr) {
    MicroPrintf("FC_DIAG user_data_null");
    return kTfLiteError;
  }

  const TfLiteEvalTensor* input =
      tflite::micro::GetEvalInput(context, node, kFullyConnectedInputTensor);
  const TfLiteEvalTensor* filter =
      tflite::micro::GetEvalInput(context, node, kFullyConnectedWeightsTensor);
  const TfLiteEvalTensor* bias =
      tflite::micro::GetEvalInput(context, node, kFullyConnectedBiasTensor);
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kFullyConnectedOutputTensor);

  MicroPrintf("FC_TENSOR_DIAG idx=%d,%d,%d,%d",
              node->inputs->data[kFullyConnectedInputTensor],
              node->inputs->data[kFullyConnectedWeightsTensor],
              node->inputs->data[kFullyConnectedBiasTensor],
              node->outputs->data[kFullyConnectedOutputTensor]);
  MicroPrintf("FC_TENSOR_DIAG ptr=%x,%x,%x,%x",
              static_cast<unsigned>(reinterpret_cast<uintptr_t>(input)),
              static_cast<unsigned>(reinterpret_cast<uintptr_t>(filter)),
              static_cast<unsigned>(reinterpret_cast<uintptr_t>(bias)),
              static_cast<unsigned>(reinterpret_cast<uintptr_t>(output)));
  MicroPrintf("FC_TENSOR_DIAG type=%d,%d,%d,%d",
              input != nullptr ? input->type : -1,
              filter != nullptr ? filter->type : -1,
              bias != nullptr ? bias->type : -1,
              output != nullptr ? output->type : -1);

  const auto& node_data = *node_data_ptr;
  const auto& data = node_data.op_data;

  long long start_time = esp_timer_get_time();
  // Checks in Prepare ensure input, output and filter types are all the same.
  switch (input->type) {
    case kTfLiteFloat32: {
      tflite::reference_ops::FullyConnected(
          FullyConnectedParamsFloat(params->activation),
          tflite::micro::GetTensorShape(input),
          tflite::micro::GetTensorData<float>(input),
          tflite::micro::GetTensorShape(filter),
          tflite::micro::GetTensorData<float>(filter),
          tflite::micro::GetTensorShape(bias),
          tflite::micro::GetTensorData<float>(bias),
          tflite::micro::GetTensorShape(output),
          tflite::micro::GetTensorData<float>(output));
      break;
    }

    case kTfLiteInt8: {
      const int32_t* bias_data =
          nullptr != bias ? tflite::micro::GetTensorData<int32_t>(bias)
                          : nullptr;
      if (node_data.per_channel_output_multiplier != nullptr) {
        const FullyConnectedParams quantized_params =
            FullyConnectedParamsQuantized(data);
        static bool quant_diag_printed = false;
        if (!quant_diag_printed) {
          int32_t multiplier_min = node_data.per_channel_output_multiplier[0];
          int32_t multiplier_max = multiplier_min;
          int shift_min = node_data.per_channel_output_shift[0];
          int shift_max = shift_min;
          for (int i = 1; i < node_data.output_depth; ++i) {
            multiplier_min = std::min(
                multiplier_min, node_data.per_channel_output_multiplier[i]);
            multiplier_max = std::max(
                multiplier_max, node_data.per_channel_output_multiplier[i]);
            shift_min =
                std::min(shift_min, node_data.per_channel_output_shift[i]);
            shift_max =
                std::max(shift_max, node_data.per_channel_output_shift[i]);
          }
          MicroPrintf(
              "FC_QUANT_DIAG depth=%d mult_hash=%x shift_hash=%x",
              node_data.output_depth,
              static_cast<unsigned>(Fnv1a32(
                  node_data.per_channel_output_multiplier,
                  node_data.output_depth * sizeof(int32_t))),
              static_cast<unsigned>(Fnv1a32(
                  node_data.per_channel_output_shift,
                  node_data.output_depth * sizeof(int))));
          MicroPrintf(
              "FC_QUANT_DIAG mult=%d:%d first=%d last=%d shift=%d:%d first=%d last=%d",
              multiplier_min, multiplier_max,
              node_data.per_channel_output_multiplier[0],
              node_data.per_channel_output_multiplier[node_data.output_depth - 1],
              shift_min, shift_max, node_data.per_channel_output_shift[0],
              node_data.per_channel_output_shift[node_data.output_depth - 1]);
          MicroPrintf(
              "FC_QUANT_DIAG input_offset=%d weights_offset=%d output_offset=%d activation=%d:%d",
              quantized_params.input_offset, quantized_params.weights_offset,
              quantized_params.output_offset,
              quantized_params.quantized_activation_min,
              quantized_params.quantized_activation_max);
#if TFLITE_SINGLE_ROUNDING
          MicroPrintf("FC_QUANT_DIAG single_rounding=1 int32_bytes=%d int_bytes=%d",
                      static_cast<int>(sizeof(int32_t)),
                      static_cast<int>(sizeof(int)));
#else
          MicroPrintf("FC_QUANT_DIAG single_rounding=0 int32_bytes=%d int_bytes=%d",
                      static_cast<int>(sizeof(int32_t)),
                      static_cast<int>(sizeof(int)));
#endif
          quant_diag_printed = true;
        }
        tflite::reference_integer_ops::FullyConnectedPerChannel(
            quantized_params,
            node_data.per_channel_output_multiplier,
            node_data.per_channel_output_shift,
            tflite::micro::GetTensorShape(input),
            tflite::micro::GetTensorData<int8_t>(input),
            tflite::micro::GetTensorShape(filter),
            tflite::micro::GetTensorData<int8_t>(filter),
            tflite::micro::GetTensorShape(bias), bias_data,
            tflite::micro::GetTensorShape(output),
            tflite::micro::GetTensorData<int8_t>(output));
        break;
      }
#if ESP_NN
      const RuntimeShape& filter_shape = tflite::micro::GetTensorShape(filter);
      const RuntimeShape& output_shape = tflite::micro::GetTensorShape(output);
      const int filter_dim_count = filter_shape.DimensionsCount();
      const int batches = output_shape.Dims(0);
      const int output_depth = output_shape.Dims(1);
      TFLITE_DCHECK_LE(output_depth, filter_shape.Dims(filter_dim_count - 2));
      const int accum_depth = filter_shape.Dims(filter_dim_count - 1);

      const int8_t *input_data = tflite::micro::GetTensorData<int8_t>(input);
      int8_t *output_data = tflite::micro::GetTensorData<int8_t>(output);
      const int8_t *filter_data = tflite::micro::GetTensorData<int8_t>(filter);

      for (int b = 0; b < batches; ++b) {
        esp_nn_fully_connected_s8(input_data, -data.input_zero_point,
                                  accum_depth,
                                  filter_data, -data.filter_zero_point,
                                  bias_data, output_data, output_depth,
                                  data.output_zero_point,
                                  data.output_shift, data.output_multiplier,
                                  data.output_activation_min,
                                  data.output_activation_max);
        input_data += accum_depth;
        output_data += output_depth;
      }
#else
      tflite::reference_integer_ops::FullyConnected(
          FullyConnectedParamsQuantized(data),
          tflite::micro::GetTensorShape(input),
          tflite::micro::GetTensorData<int8_t>(input),
          tflite::micro::GetTensorShape(filter),
          tflite::micro::GetTensorData<int8_t>(filter),
          tflite::micro::GetTensorShape(bias), bias_data,
          tflite::micro::GetTensorShape(output),
          tflite::micro::GetTensorData<int8_t>(output));
#endif
      break;
    }

    case kTfLiteUInt8: {
      tflite::reference_ops::FullyConnected(
          FullyConnectedParamsQuantized(data),
          tflite::micro::GetTensorShape(input),
          tflite::micro::GetTensorData<uint8_t>(input),
          tflite::micro::GetTensorShape(filter),
          tflite::micro::GetTensorData<uint8_t>(filter),
          tflite::micro::GetTensorShape(bias),
          tflite::micro::GetTensorData<int32_t>(bias),
          tflite::micro::GetTensorShape(output),
          tflite::micro::GetTensorData<uint8_t>(output));
      break;
    }
    default: {
      TF_LITE_KERNEL_LOG(context, "Type %s (%d) not supported.",
                         TfLiteTypeGetName(input->type), input->type);
      return kTfLiteError;
    }
  }
  fc_total_time += esp_timer_get_time() - start_time;
  return kTfLiteOk;
}

}  // namespace

TfLiteRegistration_V1 Register_FULLY_CONNECTED() {
  return tflite::micro::RegisterOp(Init, Prepare, Eval);
}

}  // namespace tflite
