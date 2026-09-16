/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/micro/kernels/conv.h"

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/reference/conv.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/conv.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/padding.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/micro_log.h"

#include <esp_timer.h>

#if ESP_NN
#include <esp_nn.h>
#endif


long long conv_total_time = 0;

namespace tflite {
namespace {

struct NodeData {
  OpDataConv op_data;
#if ESP_NN
  int buffer_idx;
  int sub_buf_idx;
#endif
};

uint32_t Fnv1a(const void* data, size_t size) {
  const auto* bytes = static_cast<const uint8_t*>(data);
  uint32_t hash = 2166136261u;
  for (size_t i = 0; i < size; ++i) {
    hash = (hash ^ bytes[i]) * 16777619u;
  }
  return hash;
}

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  TFLITE_DCHECK(context->AllocatePersistentBuffer != nullptr);
  NodeData* data = static_cast<NodeData*>(
      context->AllocatePersistentBuffer(context, sizeof(NodeData)));
#if ESP_NN
  data->buffer_idx = -1;
  data->sub_buf_idx = -1;
#endif
  return data;
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(node->user_data != nullptr);
  TFLITE_DCHECK(node->builtin_data != nullptr);

  NodeData* data = static_cast<NodeData*>(node->user_data);
  const auto& params =
      *(static_cast<const TfLiteConvParams*>(node->builtin_data));

  MicroContext* micro_context = GetMicroContext(context);

  TfLiteTensor* input =
      micro_context->AllocateTempInputTensor(node, kConvInputTensor);
  TF_LITE_ENSURE(context, input != nullptr);
  TfLiteTensor* filter =
      micro_context->AllocateTempInputTensor(node, kConvWeightsTensor);
  TF_LITE_ENSURE(context, filter != nullptr);
  TfLiteTensor* output =
      micro_context->AllocateTempOutputTensor(node, kConvOutputTensor);
  TF_LITE_ENSURE(context, output != nullptr);

  const int input_width = input->dims->data[2];
  const int input_height = input->dims->data[1];
  const int filter_width = filter->dims->data[2];
  const int filter_height = filter->dims->data[1];
  const int output_width = output->dims->data[2];
  const int output_height = output->dims->data[1];

  // Dynamically allocate per-channel quantization parameters.
  const int num_channels = filter->dims->data[kConvQuantizedDimension];
  data->op_data.per_channel_output_multiplier =
      static_cast<int32_t*>(context->AllocatePersistentBuffer(
          context, num_channels * sizeof(int32_t)));
  data->op_data.per_channel_output_shift =
      static_cast<int32_t*>(context->AllocatePersistentBuffer(
          context, num_channels * sizeof(int32_t)));

  // All per-channel quantized tensors need valid zero point and scale arrays.
  if (input->type == kTfLiteInt8) {
    TF_LITE_ENSURE_EQ(context, filter->quantization.type,
                      kTfLiteAffineQuantization);

    const auto* affine_quantization =
        static_cast<TfLiteAffineQuantization*>(filter->quantization.params);
    TFLITE_DCHECK(affine_quantization != nullptr);
    TFLITE_DCHECK(affine_quantization->scale != nullptr);
    TFLITE_DCHECK(affine_quantization->zero_point != nullptr);

    TF_LITE_ENSURE(context,
                   affine_quantization->scale->size == 1 ||
                       affine_quantization->scale->size ==
                           filter->dims->data[kConvQuantizedDimension]);
    TF_LITE_ENSURE_EQ(context, affine_quantization->scale->size,
                      affine_quantization->zero_point->size);
  }

  TF_LITE_ENSURE_STATUS(CalculateOpDataConv(
      context, node, params, input_width, input_height, filter_width,
      filter_height, output_width, output_height, input->type, &data->op_data));

#if ESP_NN
  if (input->type == kTfLiteInt8) {
    data_dims_t input_dims =  {
                                .width = input_width, .height = input_height,
                                .channels = input->dims->data[3], 1
                              };
    data_dims_t output_dims = {
                                .width = output_width, .height = output_height,
                                .channels = output->dims->data[3], 1
                              };
    data_dims_t filter_dims = {.width = filter_width, .height = filter_height, 0, 0};
    conv_params_t conv_params = {
                                  .in_offset = 0, .out_offset = 0,
                                  .stride = {params.stride_width, params.stride_height},
                                  .padding = {data->op_data.padding.width, data->op_data.padding.height},
                                  .dilation = {0, 0}, .activation = {-128, 127}
                                };

    int scratch_buf_size = esp_nn_get_conv_scratch_size(
        &input_dims, &filter_dims, &output_dims, &conv_params);
    if (scratch_buf_size > 0) {
      TF_LITE_ENSURE_STATUS(context->RequestScratchBufferInArena(
        context, scratch_buf_size, &data->buffer_idx));
    } else {
      data->buffer_idx = -1;
    }
    data->sub_buf_idx = -1;
    if ((params.dilation_width_factor != 1 ||
         params.dilation_height_factor != 1) &&
        params.dilation_width_factor == params.dilation_height_factor &&
        params.stride_width == 1 && params.stride_height == 1 &&
        filter_width == 3 && filter_height == 3 &&
        input_width % params.dilation_width_factor == 0 &&
        input_height % params.dilation_height_factor == 0) {
      const int dil = params.dilation_width_factor;
      const int sub_w = input_width / dil;
      const int sub_h = input_height / dil;
      const size_t sub_in_bytes =
          static_cast<size_t>(sub_w) * sub_h * input->dims->data[3];
      const size_t sub_out_bytes =
          static_cast<size_t>(sub_w) * sub_h * output->dims->data[3];
      TF_LITE_ENSURE_STATUS(context->RequestScratchBufferInArena(
          context, sub_in_bytes + sub_out_bytes, &data->sub_buf_idx));
    }
  }
#endif

  micro_context->DeallocateTempTfLiteTensor(output);
  micro_context->DeallocateTempTfLiteTensor(input);
  micro_context->DeallocateTempTfLiteTensor(filter);

  return kTfLiteOk;
}

#if ESP_NN
// Dilated per-channel-quantization Int8 convolution via phase sub-grids.
// A dilated 3x3 conv (rate d, stride 1, same padding) equals d^2 regular
// 3x3 same-padding convs applied to the d x d phase sub-grids, one per
// (py, px) with py = y % d, px = x % d.
inline void EvalQuantizedPerChannelDilated(
    TfLiteContext* context, TfLiteNode* node, const TfLiteConvParams& params,
    const NodeData& data, const TfLiteEvalTensor* input,
    const TfLiteEvalTensor* filter, const TfLiteEvalTensor* bias,
    TfLiteEvalTensor* output) {
  const int dil_w = params.dilation_width_factor;
  const int dil_h = params.dilation_height_factor;

  RuntimeShape input_shape = tflite::micro::GetTensorShape(input);
  RuntimeShape filter_shape = tflite::micro::GetTensorShape(filter);
  RuntimeShape output_shape = tflite::micro::GetTensorShape(output);

  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const int input_depth = input_shape.Dims(3);
  const int filter_height = filter_shape.Dims(1);
  const int filter_width = filter_shape.Dims(2);
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);
  const int output_depth = output_shape.Dims(3);

  const bool sub_grid_ok =
      dil_w == dil_h && params.stride_width == 1 &&
      params.stride_height == 1 && filter_width == 3 && filter_height == 3 &&
      input_width % dil_w == 0 && input_height % dil_h == 0 &&
      data.sub_buf_idx > -1;

  if (!sub_grid_ok) {
    reference_integer_ops::ConvPerChannel(
        ConvParamsQuantized(params, data.op_data),
        data.op_data.per_channel_output_multiplier,
        data.op_data.per_channel_output_shift, input_shape,
        tflite::micro::GetTensorData<int8_t>(input), filter_shape,
        tflite::micro::GetTensorData<int8_t>(filter),
        tflite::micro::GetTensorShape(bias),
        tflite::micro::GetTensorData<int32_t>(bias), output_shape,
        tflite::micro::GetTensorData<int8_t>(output));
    return;
  }

  const int d = dil_w;
  const int sub_w = input_width / d;
  const int sub_h = input_height / d;

  const int8_t* input_data = tflite::micro::GetTensorData<int8_t>(input);
  const int8_t* filter_data = tflite::micro::GetTensorData<int8_t>(filter);
  const int32_t* bias_data = tflite::micro::GetTensorData<int32_t>(bias);
  int8_t* output_data = tflite::micro::GetTensorData<int8_t>(output);

  const int32_t input_offset = -data.op_data.input_zero_point;
  const int32_t output_offset = data.op_data.output_zero_point;

  void* scratch_buf = nullptr;
  if (data.buffer_idx > -1) {
    scratch_buf = context->GetScratchBuffer(context, data.buffer_idx);
  }
  esp_nn_set_conv_scratch_buf(scratch_buf);

  uint8_t* sub_buf = static_cast<uint8_t*>(
      context->GetScratchBuffer(context, data.sub_buf_idx));
  int8_t* sub_in = reinterpret_cast<int8_t*>(sub_buf);
  int8_t* sub_out = sub_in + sub_w * sub_h * input_depth;

  data_dims_t sub_input_dims = {
      .width = static_cast<uint16_t>(sub_w),
      .height = static_cast<uint16_t>(sub_h),
      .channels = static_cast<uint16_t>(input_depth), 1};
  data_dims_t sub_output_dims = {
      .width = static_cast<uint16_t>(sub_w),
      .height = static_cast<uint16_t>(sub_h),
      .channels = static_cast<uint16_t>(output_depth), 1};
  data_dims_t filter_dims = {.width = filter_width, .height = filter_height, 0, 0};
  conv_params_t sub_conv_params = {
      .in_offset = input_offset, .out_offset = output_offset,
      .stride = {1, 1}, .padding = {1, 1}, .dilation = {0, 0},
      .activation = {data.op_data.output_activation_min,
                     data.op_data.output_activation_max}};
  quant_data_t quant_data = {
      .shift = data.op_data.per_channel_output_shift,
      .mult = data.op_data.per_channel_output_multiplier};

  for (int py = 0; py < d; ++py) {
    for (int px = 0; px < d; ++px) {
      for (int r = 0; r < sub_h; ++r) {
        const int iy = py + r * d;
        const int8_t* src_row =
            input_data + (iy * input_width + px) * input_depth;
        int8_t* dst_row = sub_in + r * sub_w * input_depth;
        for (int t = 0; t < sub_w; ++t) {
          memcpy(dst_row + t * input_depth, src_row + t * d * input_depth,
                 static_cast<size_t>(input_depth));
        }
      }
      esp_nn_conv_s8(&sub_input_dims, sub_in, &filter_dims, filter_data,
                     bias_data, &sub_output_dims, sub_out, &sub_conv_params,
                     &quant_data);
      for (int r = 0; r < sub_h; ++r) {
        const int oy = py + r * d;
        int8_t* dst_row =
            output_data + (oy * output_width + px) * output_depth;
        const int8_t* src_row = sub_out + r * sub_w * output_depth;
        for (int t = 0; t < sub_w; ++t) {
          memcpy(dst_row + t * d * output_depth, src_row + t * output_depth,
                 static_cast<size_t>(output_depth));
        }
      }
    }
  }
}

// Fixed-point per-channel-quantization convolution Int8 function wrapper.
inline void EvalQuantizedPerChannel(
    TfLiteContext* context, TfLiteNode* node, const TfLiteConvParams& params,
    const NodeData& data, const TfLiteEvalTensor* input,
    const TfLiteEvalTensor* filter, const TfLiteEvalTensor* bias,
    TfLiteEvalTensor* output) {
  const int dilation_width_factor = params.dilation_width_factor;
  const int dilation_height_factor = params.dilation_height_factor;

  if (dilation_width_factor == 1 && dilation_height_factor == 1) {
    // Get parameters.
    RuntimeShape filter_shape = tflite::micro::GetTensorShape(filter);
    RuntimeShape input_shape = tflite::micro::GetTensorShape(input);
    RuntimeShape output_shape = tflite::micro::GetTensorShape(output);
    RuntimeShape bias_shape = tflite::micro::GetTensorShape(bias);

    const int8_t *input_data = tflite::micro::GetTensorData<int8_t>(input);
    int8_t *output_data = tflite::micro::GetTensorData<int8_t>(output);

    const int32_t input_offset = -data.op_data.input_zero_point;
    const int32_t output_offset = data.op_data.output_zero_point;
    const int stride_width = params.stride_width;
    const int stride_height = params.stride_height;
    const int pad_width = data.op_data.padding.width;
    const int pad_height = data.op_data.padding.height;

    const int input_height = input_shape.Dims(1);
    const int input_width = input_shape.Dims(2);
    const int filter_height = filter_shape.Dims(1);
    const int filter_width = filter_shape.Dims(2);
    const int output_height = output_shape.Dims(1);
    const int output_width = output_shape.Dims(2);

    // Set min and max value of the output.
    const int32_t activation_min = data.op_data.output_activation_min;
    const int32_t activation_max = data.op_data.output_activation_max;

    // Consistency check.
    TFLITE_DCHECK_LE(activation_min, activation_max);
    TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
    TFLITE_DCHECK_EQ(filter_shape.DimensionsCount(), 4);
    TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);
    const int batch_size = MatchingDim(input_shape, 0, output_shape, 0);
    const int input_depth = MatchingDim(input_shape, 3, filter_shape, 3);
    const int output_depth = MatchingDim(filter_shape, 0, output_shape, 3);

    if (tflite::micro::GetTensorData<int8_t>(bias)) {
      TFLITE_DCHECK_EQ(bias_shape.FlatSize(), output_depth);
    }

    void *scratch_buf = NULL;
    if (data.buffer_idx > -1) {
      scratch_buf = context->GetScratchBuffer(context, data.buffer_idx);
    }
    esp_nn_set_conv_scratch_buf(scratch_buf);

    const int input_size = input_width * input_height * input_depth;
    const int output_size = output_width * output_height * output_depth;

    data_dims_t input_dims =  {
                                .width = input_width, .height = input_height,
                                .channels = input_depth, 1
                              };
    data_dims_t output_dims = {
                                .width = output_width, .height = output_height,
                                .channels = output_depth, 1
                              };
    data_dims_t filter_dims = {.width = filter_width, .height = filter_height, 0, 0};
    conv_params_t conv_params = {
                                  .in_offset = input_offset, .out_offset = output_offset,
                                  .stride = {stride_width, stride_height},
                                  .padding = {pad_width, pad_height},
                                  .dilation = {0, 0},
                                  .activation = {activation_min, activation_max}
                                };
    quant_data_t quant_data = {
                                .shift = data.op_data.per_channel_output_shift,
                                .mult = data.op_data.per_channel_output_multiplier
                              };

    for (int i_batch = 0; i_batch < batch_size; i_batch++) {
      esp_nn_conv_s8(&input_dims, input_data + i_batch * input_size,
                     &filter_dims, tflite::micro::GetTensorData<int8_t>(filter),
                     tflite::micro::GetTensorData<int32_t>(bias),
                     &output_dims, output_data + i_batch * output_size,
                     &conv_params, &quant_data);
    }
  } else {
    EvalQuantizedPerChannelDilated(context, node, params, data, input, filter,
                                   bias, output);
  }
}
#endif

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteEvalTensor* input =
      tflite::micro::GetEvalInput(context, node, kConvInputTensor);
  const TfLiteEvalTensor* filter =
      tflite::micro::GetEvalInput(context, node, kConvWeightsTensor);
  const TfLiteEvalTensor* bias =
      (NumInputs(node) == 3)
          ? tflite::micro::GetEvalInput(context, node, kConvBiasTensor)
          : nullptr;
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kConvOutputTensor);

  TFLITE_DCHECK(node->builtin_data != nullptr);
  const auto& params =
      *(reinterpret_cast<TfLiteConvParams*>(node->builtin_data));
  TFLITE_DCHECK(node->user_data != nullptr);
  const auto& data = *(static_cast<const NodeData*>(node->user_data));

  const RuntimeShape input_shape = tflite::micro::GetTensorShape(input);
  const RuntimeShape filter_shape = tflite::micro::GetTensorShape(filter);
  const RuntimeShape output_shape = tflite::micro::GetTensorShape(output);
  if (output_shape.FlatSize() == 1204224) {
    const RuntimeShape bias_shape = tflite::micro::GetTensorShape(bias);
    const auto* input_data = tflite::micro::GetTensorData<int8_t>(input);
    const auto* filter_data = tflite::micro::GetTensorData<int8_t>(filter);
    const auto* bias_data = tflite::micro::GetTensorData<int32_t>(bias);
    const auto* output_data = tflite::micro::GetTensorData<int8_t>(output);
    const int output_depth = output_shape.Dims(3);
    const uintptr_t input_start = reinterpret_cast<uintptr_t>(input_data);
    const uintptr_t filter_start = reinterpret_cast<uintptr_t>(filter_data);
    const uintptr_t bias_start = reinterpret_cast<uintptr_t>(bias_data);
    const uintptr_t output_start = reinterpret_cast<uintptr_t>(output_data);
    MicroPrintf(
        "CONV_DIAG shapes in=%dx%dx%d filter=%dx%dx%dx%d out=%dx%dx%d",
        input_shape.Dims(1), input_shape.Dims(2), input_shape.Dims(3),
        filter_shape.Dims(0), filter_shape.Dims(1), filter_shape.Dims(2),
        filter_shape.Dims(3), output_shape.Dims(1), output_shape.Dims(2),
        output_shape.Dims(3));
    MicroPrintf(
        "CONV_DIAG input=%x-%x filter=%x-%x bias=%x-%x output=%x-%x",
        static_cast<unsigned>(input_start),
        static_cast<unsigned>(input_start + input_shape.FlatSize()),
        static_cast<unsigned>(filter_start),
        static_cast<unsigned>(filter_start + filter_shape.FlatSize()),
        static_cast<unsigned>(bias_start),
        static_cast<unsigned>(bias_start +
                              bias_shape.FlatSize() * sizeof(int32_t)),
        static_cast<unsigned>(output_start),
        static_cast<unsigned>(output_start + output_shape.FlatSize()));
    MicroPrintf(
        "CONV_DIAG stride=%dx%d dilation=%dx%d pad=%dx%d in_zp=%d out_zp=%d act=%d:%d",
        params.stride_height, params.stride_width,
        params.dilation_height_factor, params.dilation_width_factor,
        data.op_data.padding.height, data.op_data.padding.width,
        data.op_data.input_zero_point, data.op_data.output_zero_point,
        data.op_data.output_activation_min, data.op_data.output_activation_max);
    MicroPrintf(
        "CONV_DIAG input_hash=%x filter_hash=%x mult_hash=%x shift_hash=%x bias_hash=%x mult=%d:%d shift=%d:%d bias=%d:%d",
        Fnv1a(input_data, input_shape.FlatSize()),
        Fnv1a(filter_data, filter_shape.FlatSize()),
        Fnv1a(data.op_data.per_channel_output_multiplier,
              output_depth * sizeof(int32_t)),
        Fnv1a(data.op_data.per_channel_output_shift,
              output_depth * sizeof(int32_t)),
        Fnv1a(bias_data, bias_shape.FlatSize() * sizeof(int32_t)),
        data.op_data.per_channel_output_multiplier[0],
        data.op_data.per_channel_output_multiplier[output_depth - 1],
        data.op_data.per_channel_output_shift[0],
        data.op_data.per_channel_output_shift[output_depth - 1], bias_data[0],
        bias_data[output_depth - 1]);
  }

  TF_LITE_ENSURE_EQ(context, input->type, output->type);
  TF_LITE_ENSURE_MSG(context, input->type == filter->type,
                     "Hybrid models are not supported on TFLite Micro.");

  long long start_time = esp_timer_get_time();
  switch (input->type) {  // Already know in/out types are same.
    case kTfLiteFloat32: {
      tflite::reference_ops::Conv(
          ConvParamsFloat(params, data.op_data),
          tflite::micro::GetTensorShape(input),
          tflite::micro::GetTensorData<float>(input),
          tflite::micro::GetTensorShape(filter),
          tflite::micro::GetTensorData<float>(filter),
          tflite::micro::GetTensorShape(bias),
          tflite::micro::GetTensorData<float>(bias),
          tflite::micro::GetTensorShape(output),
          tflite::micro::GetTensorData<float>(output),
          tflite::micro::GetTensorShape(nullptr), nullptr);
      break;
    }
    case kTfLiteInt8: {
#if ESP_NN
      EvalQuantizedPerChannel(context, node, params, data, input, filter,
                              bias, output);
#else
      reference_integer_ops::ConvPerChannel(
          ConvParamsQuantized(params, data.op_data),
          data.op_data.per_channel_output_multiplier,
          data.op_data.per_channel_output_shift,
          tflite::micro::GetTensorShape(input),
          tflite::micro::GetTensorData<int8_t>(input),
          tflite::micro::GetTensorShape(filter),
          tflite::micro::GetTensorData<int8_t>(filter),
          tflite::micro::GetTensorShape(bias),
          tflite::micro::GetTensorData<int32_t>(bias),
          tflite::micro::GetTensorShape(output),
          tflite::micro::GetTensorData<int8_t>(output));
#endif
      break;
    }
    case kTfLiteUInt8: {
      //EvalQuantized
      reference_ops::Conv(ConvParamsQuantized(params, data.op_data),
                          tflite::micro::GetTensorShape(input),
                          tflite::micro::GetTensorData<uint8_t>(input),
                          tflite::micro::GetTensorShape(filter),
                          tflite::micro::GetTensorData<uint8_t>(filter),
                          tflite::micro::GetTensorShape(bias),
                          tflite::micro::GetTensorData<int32_t>(bias),
                          tflite::micro::GetTensorShape(output),
                          tflite::micro::GetTensorData<uint8_t>(output),
                          tflite::micro::GetTensorShape(nullptr), nullptr,
                          nullptr);
      break;
    }
    default:
      TF_LITE_KERNEL_LOG(context, "Type %s (%d) not supported.",
                         TfLiteTypeGetName(input->type), input->type);
      return kTfLiteError;
  }
  long long time_this_instance = esp_timer_get_time() - start_time;
  conv_total_time += time_this_instance;
  // printf("time this conv instance: %llu\n", time_this_instance / 1000);
  return kTfLiteOk;
}

}  // namespace

TfLiteRegistration_V1 Register_CONV_2D() {
  return tflite::micro::RegisterOp(Init, Prepare, Eval);
}

}  // namespace tflite
