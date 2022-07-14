#include "rocblas.h"

#define driver_operation rocblas_operation
#define char2driver_operation char2rocblas_operation
#define driver_operation_none rocblas_operation_none
#define driver_set_pointer_mode rocblas_set_pointer_mode
#define DRIVER_POINTER_MODE_HOST rocblas_pointer_mode_host
#define DRIVER_POINTER_MODE_DEVICE rocblas_pointer_mode_device
#define CHECK_DRIVER_ERROR CHECK_ROCBLAS_ERROR
#define driver_half rocblas_half
#define driver_bfloat16 rocblas_bfloat16
#define driver_local_handle rocblas_local_handle
#define driver_type rocblas_datatype
#define driver_type2string rocblas_datatype2string
#define string2driver_type string2rocblas_datatype
#define driver_algo rocblas_gemm_algo
#define driver_stride rocblas_stride

/* ============================================================================================ */
/*  Convert rocblas constants to lapack char. */

/*! \brief  generate a random number in range [1,2,3,4,5,6,7,8,9,10] */
template <typename T>
inline T random_generator()
{
    return std::uniform_int_distribution<int>(1, 10)(rocblas_rng);
}

// for rocblas_float_complex, generate two random ints (same behaviour as for floats)
template <>
inline rocblas_float_complex random_generator<rocblas_float_complex>()
{
    return {float(std::uniform_int_distribution<int>(1, 10)(rocblas_rng)),
            float(std::uniform_int_distribution<int>(1, 10)(rocblas_rng))};
};

// for rocblas_double_complex, generate two random ints (same behaviour as for doubles)
template <>
inline rocblas_double_complex random_generator<rocblas_double_complex>()
{
    return {double(std::uniform_int_distribution<int>(1, 10)(rocblas_rng)),
            double(std::uniform_int_distribution<int>(1, 10)(rocblas_rng))};
};

// for rocblas_half, generate float, and convert to rocblas_half
/*! \brief  generate a random number in range [-2,-1,0,1,2] */
template <>
inline rocblas_half random_generator<rocblas_half>()
{
    return rocblas_half(std::uniform_int_distribution<int>(-2, 2)(rocblas_rng));
};

// for rocblas_bfloat16, generate float, and convert to rocblas_bfloat16
/*! \brief  generate a random number in range [-2,-1,0,1,2] */
template <>
inline rocblas_bfloat16 random_generator<rocblas_bfloat16>()
{
    return rocblas_bfloat16(std::uniform_int_distribution<int>(-2, 2)(rocblas_rng));
};

/*! \brief  generate a random number in range [1,2,3] */
template <>
inline int8_t random_generator<int8_t>()
{
    return std::uniform_int_distribution<int8_t>(1, 3)(rocblas_rng);
};

class rocblas_local_handle
{
    rocblas_handle handle;

public:
    rocblas_local_handle()
    {
        rocblas_create_handle(&handle);
    }
    ~rocblas_local_handle()
    {
        rocblas_destroy_handle(handle);
    }

    // Allow rocblas_local_handle to be used anywhere rocblas_handle is expected
    operator rocblas_handle&()
    {
        return handle;
    }
    operator const rocblas_handle&() const
    {
        return handle;
    }
};

constexpr auto rocblas2char_operation(rocblas_operation value)
{
    switch(value)
    {
    case rocblas_operation_none:
        return 'N';
    case rocblas_operation_transpose:
        return 'T';
    case rocblas_operation_conjugate_transpose:
        return 'C';
    }
    return '\0';
}

constexpr auto rocblas2char_fill(rocblas_fill value)
{
    switch(value)
    {
    case rocblas_fill_upper:
        return 'U';
    case rocblas_fill_lower:
        return 'L';
    case rocblas_fill_full:
        return 'F';
    }
    return '\0';
}

constexpr auto rocblas2char_diagonal(rocblas_diagonal value)
{
    switch(value)
    {
    case rocblas_diagonal_unit:
        return 'U';
    case rocblas_diagonal_non_unit:
        return 'N';
    }
    return '\0';
}

constexpr auto rocblas2char_side(rocblas_side value)
{
    switch(value)
    {
    case rocblas_side_left:
        return 'L';
    case rocblas_side_right:
        return 'R';
    case rocblas_side_both:
        return 'B';
    }
    return '\0';
}

// return precision string for rocblas_datatype
constexpr auto rocblas_datatype2string(rocblas_datatype type)
{
    switch(type)
    {
    case rocblas_datatype_f16_r:
        return "f16_r";
    case rocblas_datatype_f32_r:
        return "f32_r";
    case rocblas_datatype_f64_r:
        return "f64_r";
    case rocblas_datatype_f16_c:
        return "f16_k";
    case rocblas_datatype_f32_c:
        return "f32_c";
    case rocblas_datatype_f64_c:
        return "f64_c";
    case rocblas_datatype_i8_r:
        return "i8_r";
    case rocblas_datatype_u8_r:
        return "u8_r";
    case rocblas_datatype_i32_r:
        return "i32_r";
    case rocblas_datatype_u32_r:
        return "u32_r";
    case rocblas_datatype_i8_c:
        return "i8_c";
    case rocblas_datatype_u8_c:
        return "u8_c";
    case rocblas_datatype_i32_c:
        return "i32_c";
    case rocblas_datatype_u32_c:
        return "u32_c";
    case rocblas_datatype_bf16_r:
        return "bf16_r";
    case rocblas_datatype_bf16_c:
        return "bf16_c";
    default:
        return "invalid";
    }
}

// constexpr auto rocblas_initialization2string(rocblas_initialization init)
// {
//     switch(init)
//     {
//     case rocblas_initialization_random_int:
//         return "rand_int";
//     case rocblas_initialization_random_narrow:
//         return "rand_narrow";
//     case rocblas_initialization_random_broad:
//         return "rand_broad";
//     case rocblas_initialization_random_full:
//         return "rand_full";
//     case rocblas_initialization_trig_float:
//         return "trig_float";
//     case rocblas_initialization_hpl:
//         return "hpl";
//     case rocblas_initialization_const:
//         return "const";
//     case rocblas_initialization_file:
//         return "file";
//     default:
//         return "invalid";
//     }
// }

/* ============================================================================================ */
/*  Convert lapack char constants to rocblas type. */

// Absolute value
template <typename T, typename std::enable_if<!is_complex<T>, int>::type = 0>
__device__ __host__ inline T rocblas_abs(T x)
{
    return x < 0 ? -x : x;
}

// For complex, we have defined a __device__ __host__ compatible std::abs
template <typename T, typename std::enable_if<is_complex<T>, int>::type = 0>
__device__ __host__ inline auto rocblas_abs(T x)
{
    return std::abs(x);
}

// driver_half
__device__ __host__ inline driver_half rocblas_abs(driver_half x)
{
    union
    {
        driver_half x;
        uint16_t     data;
    } t = {x};
    t.data &= 0x7fff;
    return t.x;
}

// driver_bfloat16 is handled specially
__device__ __host__ inline driver_bfloat16 rocblas_abs(driver_bfloat16 x)
{
    x.data &= 0x7fff;
    return x;
}

// // Output driver_half value
// inline std::ostream& operator<<(std::ostream& os, driver_half x)
// {
//     return os << float(x);
// }

template <typename T>
void normalizeInputs(driver_operation transa,
                     driver_operation transb,
                     size_t            m,
                     size_t            n,
                     size_t            k,
                      std::vector<T>& a,
                     size_t lda,
                     int64_t stride_a,
                      std::vector<T>& b,
                     size_t ldb,
                     int64_t stride_b,
                     size_t batch)
{
    // We divide each element of B by the maximum corresponding element of A such that elem(A * B) <
    // 2 ** NSIGN
    if(transa == driver_operation_none)
    {
        for(size_t i = 0; i < batch; i++)
        {
            for(size_t j = 0; j < k; ++j)
            {
                T scal = T(0);
                for(size_t k = 0; k < m; ++k)
                {
                    T val = T(rocblas_abs(a[i * stride_a + j * lda + k]));
                    if(val > scal)
                        scal = val;
                }

                if(!scal)
                    abort();

                scal = T(1) / scal;
                if(transb == driver_operation_none)
                    for(size_t k = 0; k < n; ++k)
                        b[i * stride_b + j * ldb + k] *= scal;
                else
                    for(size_t k = 0; k < n; ++k)
                        b[i * stride_b + k * ldb + j] *= scal;
            }
        }
    }
    else
    {
        for(size_t i = 0; i < batch; i++)
        {
            for(size_t j = 0; j < k; ++j)
            {
                T scal = T(0);
                for(size_t k = 0; k < m; ++k)
                {
                    T val = T(rocblas_abs(a[i * stride_a + k * lda + j]));
                    if(val > scal)
                        scal = val;
                }

                if(!scal)
                    abort();

                scal = T(1) / scal;
                if(transb == driver_operation_none)
                    for(size_t k = 0; k < n; ++k)
                        b[i * stride_b + j * ldb + k] *= scal;
                else
                    for(size_t k = 0; k < n; ++k)
                        b[i * stride_b + k * ldb + j] *= scal;
            }
        }
    }
}

constexpr rocblas_operation char2rocblas_operation(char value)
{
    switch(value)
    {
    case 'N':
    case 'n':
        return rocblas_operation_none;
    case 'T':
    case 't':
        return rocblas_operation_transpose;
    case 'C':
    case 'c':
        return rocblas_operation_conjugate_transpose;
    default:
        return static_cast<rocblas_operation>(-1);
    }
}

constexpr rocblas_fill char2rocblas_fill(char value)
{
    switch(value)
    {
    case 'U':
    case 'u':
        return rocblas_fill_upper;
    case 'L':
    case 'l':
        return rocblas_fill_lower;
    default:
        return static_cast<rocblas_fill>(-1);
    }
}

constexpr rocblas_diagonal char2rocblas_diagonal(char value)
{
    switch(value)
    {
    case 'U':
    case 'u':
        return rocblas_diagonal_unit;
    case 'N':
    case 'n':
        return rocblas_diagonal_non_unit;
    default:
        return static_cast<rocblas_diagonal>(-1);
    }
}

constexpr rocblas_side char2rocblas_side(char value)
{
    switch(value)
    {
    case 'L':
    case 'l':
        return rocblas_side_left;
    case 'R':
    case 'r':
        return rocblas_side_right;
    default:
        return static_cast<rocblas_side>(-1);
    }
}

// inline rocblas_initialization string2rocblas_initialization(const std::string& value)
// {
//     // clang-format off
//     return
//         value == "rand_int"   ? rocblas_initialization_random_int :
//         value == "rand_narrow" ? rocblas_initialization_random_narrow:
//         value == "rand_broad" ? rocblas_initialization_random_broad:
//         value == "rand_full" ? rocblas_initialization_random_full:
//         value == "trig_float" ? rocblas_initialization_trig_float :
//         value == "hpl"        ? rocblas_initialization_hpl        :
//         value == "const"        ? rocblas_initialization_const        :
//         value == "file"        ? rocblas_initialization_file      :
        
//         static_cast<rocblas_initialization>(-1);
//     // clang-format on
// }

inline rocblas_datatype string2rocblas_datatype(const std::string& value)
{
    // clang-format off
    return
        value == "f16_r" || value == "h" ? rocblas_datatype_f16_r :
        value == "f32_r" || value == "s" ? rocblas_datatype_f32_r :
        value == "f64_r" || value == "d" ? rocblas_datatype_f64_r :
        value == "bf16_r"                ? rocblas_datatype_bf16_r :
        value == "f16_c"                 ? rocblas_datatype_f16_c :
        value == "f32_c" || value == "c" ? rocblas_datatype_f32_c :
        value == "f64_c" || value == "z" ? rocblas_datatype_f64_c :
        value == "bf16_c"                ? rocblas_datatype_bf16_c :
        value == "i8_r"                  ? rocblas_datatype_i8_r  :
        value == "i32_r"                 ? rocblas_datatype_i32_r :
        value == "i8_c"                  ? rocblas_datatype_i8_c  :
        value == "i32_c"                 ? rocblas_datatype_i32_c :
        value == "u8_r"                  ? rocblas_datatype_u8_r  :
        value == "u32_r"                 ? rocblas_datatype_u32_r :
        value == "u8_c"                  ? rocblas_datatype_u8_c  :
        value == "u32_c"                 ? rocblas_datatype_u32_c :
        static_cast<rocblas_datatype>(-1);
    // clang-format on
}

// clang-format off
// return letter N,T,C in place of rocblas_operation enum
constexpr char rocblas_transpose_letter(rocblas_operation trans)
{
    switch(trans)
    {
    case rocblas_operation_none:                return 'N';
    case rocblas_operation_transpose:           return 'T';
    case rocblas_operation_conjugate_transpose: return 'C';
    }
    return ' ';
}

// return letter L, R, B in place of rocblas_side enum
constexpr char rocblas_side_letter(rocblas_side side)
{
    switch(side)
    {
    case rocblas_side_left:  return 'L';
    case rocblas_side_right: return 'R';
    case rocblas_side_both:  return 'B';
    }
    return ' ';
}

// return letter U, L, B in place of rocblas_fill enum
constexpr char rocblas_fill_letter(rocblas_fill fill)
{
    switch(fill)
    {
    case rocblas_fill_upper: return 'U';
    case rocblas_fill_lower: return 'L';
    case rocblas_fill_full:  return 'F';
    }
    return ' ';
}

// return letter N, U in place of rocblas_diagonal enum
constexpr char rocblas_diag_letter(rocblas_diagonal diag)
{
    switch(diag)
    {
    case rocblas_diagonal_non_unit: return 'N';
    case rocblas_diagonal_unit:     return 'U';
    }
    return ' ';
}

// return precision string for rocblas_datatype
constexpr const char* rocblas_datatype_string(rocblas_datatype type)
{
    switch(type)
    {
    case rocblas_datatype_f16_r:  return "f16_r";
    case rocblas_datatype_f32_r:  return "f32_r";
    case rocblas_datatype_f64_r:  return "f64_r";
    case rocblas_datatype_f16_c:  return "f16_c";
    case rocblas_datatype_f32_c:  return "f32_c";
    case rocblas_datatype_f64_c:  return "f64_c";
    case rocblas_datatype_i8_r:   return "i8_r";
    case rocblas_datatype_u8_r:   return "u8_r";
    case rocblas_datatype_i32_r:  return "i32_r";
    case rocblas_datatype_u32_r:  return "u32_r";
    case rocblas_datatype_i8_c:   return "i8_c";
    case rocblas_datatype_u8_c:   return "u8_c";
    case rocblas_datatype_i32_c:  return "i32_c";
    case rocblas_datatype_u32_c:  return "u32_c";
    case rocblas_datatype_bf16_r: return "bf16_r";
    case rocblas_datatype_bf16_c: return "bf16_c";
    }
    return "invalid";
}

// return sizeof rocblas_datatype
constexpr size_t rocblas_sizeof_datatype(rocblas_datatype type)
{
    switch(type)
    {
    case rocblas_datatype_f16_r:  return 2;
    case rocblas_datatype_f32_r:  return 4;
    case rocblas_datatype_f64_r:  return 8;
    case rocblas_datatype_f16_c:  return 4;
    case rocblas_datatype_f32_c:  return 8;
    case rocblas_datatype_f64_c:  return 16;
    case rocblas_datatype_i8_r:   return 1;
    case rocblas_datatype_u8_r:   return 1;
    case rocblas_datatype_i32_r:  return 4;
    case rocblas_datatype_u32_r:  return 4;
    case rocblas_datatype_i8_c:   return 2;
    case rocblas_datatype_u8_c:   return 2;
    case rocblas_datatype_i32_c:  return 8;
    case rocblas_datatype_u32_c:  return 8;
    case rocblas_datatype_bf16_r: return 2;
    case rocblas_datatype_bf16_c: return 4;
    }
    return 0;
}

// Convert atomics mode to string
constexpr const char* rocblas_atomics_mode_to_string(rocblas_atomics_mode mode)
{
    return mode != rocblas_atomics_not_allowed ? "atomics_allowed" : "atomics_not_allowed";
}

// Convert gemm flags to string
constexpr const char* rocblas_gemm_flags_to_string(rocblas_gemm_flags)
{
    return "none";
}

// gemm
template <typename T>
static rocblas_status (*rocblas_gemm)(rocblas_handle    handle,
                               rocblas_operation transA,
                               rocblas_operation transB,
                               rocblas_int       m,
                               rocblas_int       n,
                               rocblas_int       k,
                               const T*          alpha,
                               const T*          A,
                               rocblas_int       lda,
                               const T*          B,
                               rocblas_int       ldb,
                               const T*          beta,
                               T*                C,
                               rocblas_int       ldc);

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_gemm<rocblas_half> = rocblas_hgemm;

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_gemm<float> = rocblas_sgemm;

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_gemm<double> = rocblas_dgemm;

// gemm_strided_batched
template <typename T>
static rocblas_status (*rocblas_gemm_strided_batched)(rocblas_handle    handle,
                                               rocblas_operation transA,
                                               rocblas_operation transB,
                                               rocblas_int       m,
                                               rocblas_int       n,
                                               rocblas_int       k,
                                               const T*          alpha,
                                               const T*          A,
                                               rocblas_int       lda,
                                               rocblas_stride    bsa,
                                               const T*          B,
                                               rocblas_int       ldb,
                                               rocblas_stride    bsb,
                                               const T*          beta,
                                               T*                C,
                                               rocblas_int       ldc,
                                               rocblas_stride    bsc,
                                               rocblas_int       batch_count);

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_gemm_strided_batched<rocblas_half> = rocblas_hgemm_strided_batched;

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_gemm_strided_batched<float> = rocblas_sgemm_strided_batched;

template <>
ROCBLAS_CLANG_STATIC constexpr auto rocblas_gemm_strided_batched<double> = rocblas_dgemm_strided_batched;

#define driver_gemm rocblas_gemm
#define driver_gemm_strided_batched rocblas_gemm_strided_batched
#define driver_gemm_ex rocblas_gemm_ex
// #define driver_type(X) X
