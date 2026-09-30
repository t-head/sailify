#include <cuda_runtime.h>
#ifndef AT_PER_OPERATOR_HEADERS
#else
#endif


namespace at::native {
namespace {

__device__
inline std::pair<int64_t, int64_t> get_index_mapping1d(
    int64_t input_w, int64_t output_w,
    int64_t output_x,
    int64_t pad_l) {
                         
  auto input_offset =
    (blockIdx.y + blockIdx.z * gridDim.y) * input_w;
  auto output_offset =
    (blockIdx.y + blockIdx.z * gridDim.y) * output_w;

  auto i_start_x = ::max(int64_t(0), -pad_l);
  auto o_start_x = ::max(int64_t(0), pad_l);

  int64_t input_x = ::abs(output_x - pad_l)
                    - ::abs(output_x - (input_w + pad_l - 1))
                    - output_x
                    + 2 * pad_l + input_w - 1
                    - o_start_x + i_start_x;

  return std::make_pair<int64_t, int64_t>(
    input_offset + input_x, output_offset + output_x);
}


__device__
inline std::pair<int64_t, int64_t>  get_index_mapping2d(
    int64_t input_dim_x, int64_t input_dim_y,
    int64_t output_dim_x, int64_t output_dim_y,
    int64_t pad_l, int64_t pad_t,
    int64_t output_xy, int y_shift, int z_shift, int nplane) {
                         
  auto input_offset =
    ((blockIdx.y + y_shift) + (blockIdx.z + z_shift) * nplane) * input_dim_x * input_dim_y;
  auto output_offset =
    ((blockIdx.y + y_shift) + (blockIdx.z + z_shift) * nplane) * output_dim_x * output_dim_y;

  auto output_x = output_xy % output_dim_x;
  auto output_y = output_xy / output_dim_x;

  auto i_start_x = ::max(int64_t(0), -pad_l);
  auto i_start_y = ::max(int64_t(0), -pad_t);
  auto o_start_x = ::max(int64_t(0), pad_l);
  auto o_start_y = ::max(int64_t(0), pad_t);

  auto input_x = ::abs(output_x - pad_l)
                 - ::abs(output_x - (input_dim_x + pad_l - 1))
                 - output_x
                 + 2 * pad_l + input_dim_x - 1
                 - o_start_x + i_start_x;

  auto input_y = ::abs(output_y - pad_t)
                 - ::abs(output_y - (input_dim_y + pad_t - 1))
                 - output_y
                 + 2 * pad_t + input_dim_y - 1
                 - o_start_y + i_start_y;

  return std::make_pair<int64_t, int64_t>(
    input_offset + input_y * input_dim_x + input_x,
    output_offset + output_y * output_dim_x + output_x);
}

__device__ __forceinline__ int64_t reflect_index(int64_t x, int64_t len) {
  const int64_t two = (len - 1) * 2;
  if (two <= 0) {
    return 0;
  }
  int64_t m = x % two;
  if (m < 0) m += two;
  return (m < len) ? m : (two - m);
}

template<typename scalar_t>
__global__ void reflection_pad1d_out_kernel(
    const scalar_t * input, scalar_t * output,
    int64_t input_w,
    int64_t pad_l, int64_t pad_r) {
  auto output_x = threadIdx.x + blockIdx.x * blockDim.x;
  auto output_w = input_w + pad_l + pad_r;

  if (output_x < output_w) {
    auto index_pair = get_index_mapping1d(input_w, output_w, output_x, pad_l);
    output[index_pair.second] = input[index_pair.first];
  }
}

template <typename scalar_t>
__global__ void reflection_pad1d_flat(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int64_t input_w, int64_t pad_l, int64_t pad_r,
    int64_t out_w, int64_t plane_count) {

  const int64_t bx = blockDim.x;
  const int64_t tx = threadIdx.x;

  const int64_t total = plane_count * out_w;
  const int64_t grid_stride = static_cast<int64_t>(bx) * gridDim.x;
  int64_t linear = static_cast<int64_t>(blockIdx.x) * bx + tx;

  for (; linear < total; linear += grid_stride) {
    const int64_t plane = linear / out_w;
    const int64_t x = linear - plane * out_w;
    const int64_t j = reflect_index(x - pad_l, input_w);
    output[plane * out_w + x] = input[plane * input_w + j];
  }
}

template <typename scalar_t>
__global__ void reflection_pad1d_backward_out_kernel(
    scalar_t * grad_input, const scalar_t * grad_output,
    int64_t input_w,
    int64_t pad_l, int64_t pad_r) {
  auto output_x = threadIdx.x + blockIdx.x * blockDim.x;
  auto output_w = input_w + pad_l + pad_r;

  if (output_x < output_w) {
    auto index_pair = get_index_mapping1d(input_w, output_w, output_x, pad_l);
    gpuAtomicAddNoReturn(
      &grad_input[index_pair.first], grad_output[index_pair.second]);
  }
}

template<typename scalar_t>
__global__ void reflection_pad2d_out_kernel(
    const scalar_t * input, scalar_t * output,
    int64_t input_dim_x, int64_t input_dim_y,
    int pad_t, int pad_b, int pad_l, int pad_r, int y_shift, int z_shift, int nplane) {
  auto output_xy = threadIdx.x + blockIdx.x * blockDim.x;
  auto output_dim_x = input_dim_x + pad_l + pad_r;
  auto output_dim_y = input_dim_y + pad_t + pad_b;

  if (output_xy < output_dim_x * output_dim_y) {
    auto index_pair = get_index_mapping2d(
      input_dim_x, input_dim_y,
      output_dim_x, output_dim_y,
      pad_l, pad_t,
      output_xy, y_shift, z_shift, nplane);

    output[index_pair.second] = input[index_pair.first];
  }
}

template <typename scalar_t>
__global__ void reflection_pad2d_backward_out_kernel(
    scalar_t * grad_input, const scalar_t * grad_output,
    int64_t input_dim_x, int64_t input_dim_y,
    int pad_t, int pad_b, int pad_l, int pad_r, int y_shift, int z_shift, int nplane) {
  auto output_xy = threadIdx.x + blockIdx.x * blockDim.x;
  auto output_dim_x = input_dim_x + pad_l + pad_r;
  auto output_dim_y = input_dim_y + pad_t + pad_b;

  if (output_xy < output_dim_x * output_dim_y) {
    auto index_pair = get_index_mapping2d(
      input_dim_x, input_dim_y,
      output_dim_x, output_dim_y,
      pad_l, pad_t,
      output_xy, y_shift, z_shift, nplane);

    gpuAtomicAddNoReturn(&grad_input[index_pair.first], grad_output[index_pair.second]);
  }
}

template <typename scalar_t>
__global__ void reflection_pad2d_backward_det_out_kernel(
    scalar_t* grad_input,
    const scalar_t* grad_output,
    int64_t input_dim_x,
    int64_t input_dim_y,
    int pad_t,
    int pad_b,
    int pad_l,
    int pad_r,
    int batch,
    int channels,
    int) {
  const int64_t input_xy_ = threadIdx.x + blockIdx.x * blockDim.x;
  const auto output_dim_x = input_dim_x + pad_l + pad_r;
  const auto output_dim_y = input_dim_y + pad_t + pad_b;
  const auto N = output_dim_x * output_dim_y;
  const int64_t width = output_dim_x;
  const int64_t height = output_dim_y;
  const int64_t stride =
      static_cast<int64_t>(gridDim.x) * static_cast<int64_t>(blockDim.x);
  const int64_t end =
      static_cast<int64_t>(batch) * channels * input_dim_x * input_dim_y;

  for (int64_t input_xy = input_xy_; input_xy < end; input_xy += stride) {
    scalar_t partial = 0;

    const int64_t b = input_xy / (channels * input_dim_x * input_dim_y);
    const int64_t c = (input_xy / (input_dim_x * input_dim_y)) % channels;
    const int64_t pos_xy = input_xy % (input_dim_x * input_dim_y);
    const int64_t inp_row = pos_xy / input_dim_x;
    const int64_t inp_col = pos_xy % input_dim_x;

    const bool is_top = (inp_row >= 1) && (inp_row <= pad_t);
    const bool is_bottom =
        (inp_row < input_dim_y - 1) && (inp_row >= input_dim_y - pad_b - 1);
    const bool is_left = (inp_col >= 1) && (inp_col <= pad_l);
    const bool is_right =
        (inp_col < input_dim_x - 1) && (inp_col >= input_dim_x - pad_r - 1);

    if (is_top) {
      const int64_t border_top_row = 0;
      const int64_t dist_from_t = inp_row;

      const int64_t border_top_out_row = border_top_row + pad_t;
      const int64_t border_top_out_col = pad_l + inp_col;

      const int64_t reflected_top_row = border_top_out_row - dist_from_t;
      const int64_t reflected_top_out =
          reflected_top_row * width + border_top_out_col;

      if (reflected_top_out < N) {
        partial += grad_output
            [b * (channels * width * height) + c * (width * height) +
             reflected_top_out];
      }

      if (is_left) {            
        const int64_t corner_tl_out_row = pad_t;
        const int64_t corner_tl_out_col = pad_l;
        const int64_t dist_rows = inp_row;
        const int64_t dist_cols = inp_col;
        const int64_t reflect_tl_out_row = (corner_tl_out_row - dist_rows);
        const int64_t reflect_tl_out_col = (corner_tl_out_col - dist_cols);
        const int64_t reflect_tl_out =
            (reflect_tl_out_row * width) + reflect_tl_out_col;

        if (reflect_tl_out >= 0 && reflect_tl_out < N) {
          partial += grad_output
              [b * (channels * width * height) + c * (width * height) +
               reflect_tl_out];
        }
      } else if (is_right) {             
                                          
        const int64_t corner_tr_out_row = pad_t;
        const int64_t corner_tr_out_col = pad_l + input_dim_x - 1;
        const int64_t dist_rows = inp_row;                                     
        const int64_t dist_cols = ::abs(inp_col - (input_dim_x - 1));

                                                                      
                                                                      
        const int64_t reflect_tr_out_row = (corner_tr_out_row - dist_rows);
        const int64_t reflect_tr_out_col = (corner_tr_out_col + dist_cols);
        const int64_t reflect_tr_out =
            (reflect_tr_out_row * width) + reflect_tr_out_col;

        if (reflect_tr_out >= 0 && reflect_tr_out < N) {
          partial += grad_output
              [b * (channels * width * height) + c * (width * height) +
               reflect_tr_out];
        }
      }
    }

    if (is_bottom) {
      const int64_t border_bot_row =
          input_dim_y - 1;                                  
      const int64_t border_bot_col = inp_col;
      const int64_t dist_from_bot = ::abs(inp_row - border_bot_row);

                                                                         
      const int64_t border_bot_out_row = pad_t + border_bot_row;
      const int64_t border_bot_out_col = pad_l + border_bot_col;
      const int64_t reflect_bot_row = (border_bot_out_row + dist_from_bot);
      const int64_t reflect_bot_out =
          (reflect_bot_row * width) + border_bot_out_col;

      if (reflect_bot_out >= 0 && reflect_bot_out < N) {
        partial += grad_output
            [b * (channels * width * height) + c * (width * height) +
             reflect_bot_out];
      }

      if (is_left) {
                        
        const int64_t corner_bl_row = input_dim_y - 1;
        const int64_t corner_bl_col = 0;

        const int64_t corner_bl_out_row = pad_t + corner_bl_row;
        const int64_t corner_bl_out_col = pad_l + corner_bl_col;

                                                        
        const int64_t dist_rows = ::abs(inp_row - corner_bl_row);
        const int64_t dist_cols = inp_col;

                                                                 
        const int64_t reflect_bl_out_row = (corner_bl_out_row + dist_rows);
        const int64_t reflect_bl_out_col = (corner_bl_out_col - dist_cols);
        const int64_t reflect_bl_out =
            (reflect_bl_out_row * width) + reflect_bl_out_col;

        if (reflect_bl_out >= 0 && reflect_bl_out < N) {
          partial += grad_output
              [b * (channels * width * height) + c * (width * height) +
               reflect_bl_out];
        }
      } else if (is_right) {
                           
        const int64_t corner_br_row = input_dim_y - 1;
        const int64_t corner_br_col = input_dim_x - 1;
        const int64_t dist_rows = ::abs(inp_row - corner_br_row);
        const int64_t dist_cols = ::abs(inp_col - corner_br_col);

        const int64_t corner_br_out_row = pad_t + corner_br_row;
        const int64_t corner_br_out_col = pad_l + corner_br_col;

        const int64_t reflect_br_out_row = (corner_br_out_row + dist_rows);
        const int64_t reflect_br_out_col = (corner_br_out_col + dist_cols);
        const int64_t reflect_br_out =
            (reflect_br_out_row * width) + reflect_br_out_col;

        if (reflect_br_out >= 0 && reflect_br_out < N) {
          partial += grad_output
              [b * (channels * width * height) + c * (width * height) +
               reflect_br_out];
        }
      }
    }
    if (is_left) {
      const int64_t border_left_row = inp_row;
      const int64_t border_left_out_row = border_left_row + pad_t;
      const int64_t border_left_out_col = pad_l;

      const int64_t dist_from_left = inp_col;

      const int64_t reflect_left_out_row = border_left_out_row;
      const int64_t reflect_left_out_col = border_left_out_col - dist_from_left;
      const int64_t reflect_left_out =
          reflect_left_out_row * width + reflect_left_out_col;

      if (reflect_left_out >= 0 && reflect_left_out < N) {
        partial += grad_output
            [b * (channels * width * height) + c * (width * height) +
             reflect_left_out];
      }
    }
    if (is_right) {
      const int64_t border_right_row = inp_row;
      const int64_t border_right_col = input_dim_x - 1;

      const int64_t border_right_out_row = border_right_row + pad_t;
      const int64_t border_right_out_col = border_right_col + pad_l;

      const int64_t dist_from_right = ::abs(inp_col - border_right_col);

      const int64_t reflect_right_out_row = border_right_out_row;
      const int64_t reflect_right_out_col =
          border_right_out_col + dist_from_right;
      const int64_t reflect_right_out =
          reflect_right_out_row * width + reflect_right_out_col;

      if (reflect_right_out >= 0 && reflect_right_out < N) {
        partial += grad_output
            [b * (channels * width * height) + c * (width * height) +
             reflect_right_out];
      }
    }
    const int64_t out_row = inp_row + pad_t;
    const int64_t out_col = inp_col + pad_l;

    partial += grad_output
        [b * (channels * width * height) + c * (width * height) +
         out_row * width + out_col];

    grad_input[input_xy] += partial;
  }
}

}
}