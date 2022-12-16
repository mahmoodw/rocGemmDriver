#include "hipblas.h"
// #include "cuda_bf16.h"

struct hipblasBfloat16_ex : hipblasBfloat16
{ 
    hipblasBfloat16_ex() = default;
    
    hipblasBfloat16_ex(hipblasBfloat16 val)
    {
        hipblasBfloat16 temp = val;
        data = reinterpret_cast<uint16_t&>(temp);
    }

    hipblasBfloat16_ex(double val);
};

inline hipblasBfloat16_ex float_to_bfloat16(float f)
{
    hipblasBfloat16_ex rv;
    union
    {
        float    fp32;
        uint32_t int32;
    } u = {f};
    if(~u.int32 & 0x7f800000)
    {
        u.int32 += 0x7fff + ((u.int32 >> 16) & 1); // Round to nearest, round to even
    }
    else if(u.int32 & 0xffff)
    {
        u.int32 |= 0x10000; // Preserve signaling NaN
    }
    rv.data = uint16_t(u.int32 >> 16);
    return rv;
}

inline __host__ hipblasBfloat16_ex float_to_bfloat16_truncate2(float val)
{
    union
    {
        float    fp32;
        uint32_t int32;
    } u = {val};
    hipblasBfloat16_ex ret;
    ret.data = uint16_t(u.int32 >> 16);
    if((u.int32 & 0x7fff0000) == 0x7f800000 && u.int32 & 0xffff)
        ret.data |= 1; // Preserve signaling NaN
    return ret;
}

hipblasBfloat16_ex::hipblasBfloat16_ex(double val)
{
    hipblasBfloat16 temp = float_to_bfloat16(val);
    data = reinterpret_cast<uint16_t&>(temp);
}

#define driver_operation hipblasOperation_t
#define char2driver_operation char2hipblas_operation
#define driver_operation_none HIPBLAS_OP_N
#define driver_set_pointer_mode hipblasSetPointerMode
#define DRIVER_POINTER_MODE_HOST HIPBLAS_POINTER_MODE_HOST
#define DRIVER_POINTER_MODE_DEVICE HIPBLAS_POINTER_MODE_DEVICE
#define CHECK_DRIVER_ERROR CHECK_HIPBLAS_ERROR
#define driver_half hipblasHalf
#define driver_bfloat16 hipblasBfloat16_ex
#define driver_local_handle hipblasLocalHandle
#define driver_type hipblasDatatype_t
#define driver_type2string hipblas_datatype2string
#define string2driver_type string2hipblas_datatype
#define driver_algo hipblasGemmAlgo_t
#define driver_stride hipblasStride

/*! \brief  generate a random number in HPL-like [-0.5,0.5] doubles  */
template <typename T>
inline T random_hpl_generator()
{
    return std::uniform_real_distribution<double>(-0.5, 0.5)(rocblas_rng);
}

template <>
inline driver_bfloat16 random_hpl_generator()
{
    return driver_bfloat16(float_to_bfloat16(std::uniform_real_distribution<double>(-0.5, 0.5)(rocblas_rng)));
}

/*! \brief  generate a random number in range [1,2,3,4,5,6,7,8,9,10] */
template <typename T>
inline T random_generator()
{
    return std::uniform_int_distribution<int>(1, 10)(rocblas_rng);
}

template <>
inline driver_bfloat16 random_generator<driver_bfloat16>()
{
    return driver_bfloat16(float_to_bfloat16(std::uniform_int_distribution<int>(-2, 2)(rocblas_rng))); 
};

// for hipblasHalf, generate float, and convert to hipblasHalf
/*! \brief  generate a random number in range [1,2,3] */
// template <>
// inline hipblasHalf random_generator<hipblasHalf>()
// {
//     return float_to_half(std::uniform_int_distribution<int>(-2, 2)(rocblas_rng)); 
// };

template <>
inline hipblasHalf random_generator<hipblasHalf>()
{
    driver_bfloat16 temp = float_to_bfloat16_truncate2(std::uniform_int_distribution<int>(-2, 2)(rocblas_rng));
    return reinterpret_cast<hipblasHalf&>(temp);
};

template <typename T>
static constexpr bool is_complex = false;

template <>
HIPBLAS_CLANG_STATIC constexpr bool is_complex<hipblasComplex> = true;

template <>
HIPBLAS_CLANG_STATIC constexpr bool is_complex<hipblasDoubleComplex> = true;

// Absolute value
template <typename T, typename std::enable_if<!is_complex<T>, int>::type = 0>
__device__ __host__ inline T driver_abs(T x)
{
    return x < 0 ? -x : x;
}

// For complex, we have defined a __device__ __host__ compatible std::abs
template <typename T, typename std::enable_if<is_complex<T>, int>::type = 0>
__device__ __host__ inline auto driver_abs(T x)
{
    return std::abs(x);
}

// driver_half
__device__ __host__ inline driver_half driver_abs(driver_half x)
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
__device__ __host__ inline driver_bfloat16 driver_abs(driver_bfloat16 x)
{
    x.data &= 0x7fff;
    return x;
}

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
                    T val = T(driver_abs(a[i * stride_a + j * lda + k]));
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
                    T val = T(driver_abs(a[i * stride_a + k * lda + j]));
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

// template <>
// void normalizeInputs<driver_bfloat16>(driver_operation transa,
//                      driver_operation transb,
//                      size_t            m,
//                      size_t            n,
//                      size_t            k,
//                       std::vector<driver_bfloat16>& a,
//                      size_t lda,
//                      int64_t stride_a,
//                       std::vector<driver_bfloat16>& b,
//                      size_t ldb,
//                      int64_t stride_b,
//                      size_t batch)
// {
//     // We divide each element of B by the maximum corresponding element of A such that elem(A * B) <
//     // 2 ** NSIGN
//     if(transa == driver_operation_none)
//     {
//         for(size_t i = 0; i < batch; i++)
//         {
//             for(size_t j = 0; j < k; ++j)
//             {
//                 __nv_bfloat16 scal = 0.0f;
//                 for(size_t k = 0; k < m; ++k)
//                 {
//                     driver_bfloat16 temp = driver_bfloat16(a[i * stride_a + j * lda + k]);
//                     __nv_bfloat16 val =  reinterpret_cast<__nv_bfloat16&>(temp);
//                     val = driver_abs(val);
//                     if(val > scal)
//                         scal = val;
//                 }

//                 if(!scal)
//                     abort();

//                 scal = __nv_bfloat16(1.0f) / scal;
//                 if(transb == driver_operation_none)
//                     for(size_t k = 0; k < n; ++k)
//                     {   
//                         driver_bfloat16 temp = b[i * stride_b + j * ldb + k];
//                         __nv_bfloat16 product = reinterpret_cast<__nv_bfloat16&>(temp) * scal;
//                         b[i * stride_b + j * ldb + k] = reinterpret_cast<driver_bfloat16&>(product);
//                     }
//                 else
//                     for(size_t k = 0; k < n; ++k)
//                     {   
//                         driver_bfloat16 temp = b[i * stride_b + k * ldb + j];
//                         __nv_bfloat16 product = reinterpret_cast<__nv_bfloat16&>(temp) * scal;
//                         b[i * stride_b + k * ldb + j] = reinterpret_cast<driver_bfloat16&>(product);
//                     }
//             }
//         }
//     }
//     else
//     {
//         for(size_t i = 0; i < batch; i++)
//         {
//             for(size_t j = 0; j < k; ++j)
//             {
//                 __nv_bfloat16 scal = 0.0f;
//                 for(size_t k = 0; k < m; ++k)
//                 {
//                     driver_bfloat16 temp = driver_bfloat16(a[i * stride_a + k * lda + j]);
//                     __nv_bfloat16 val = reinterpret_cast<__nv_bfloat16&>(temp);
//                     val = driver_abs(val);
//                     if(val > scal)
//                         scal = val;
//                 }

//                 if(!scal)
//                     abort();

//                 scal = __nv_bfloat16(1.0f) / scal;
//                 if(transb == driver_operation_none)
//                     for(size_t k = 0; k < n; ++k)
//                     {   
//                         driver_bfloat16 temp = b[i * stride_b + j * ldb + k];
//                         __nv_bfloat16 product = reinterpret_cast<__nv_bfloat16&>(temp) * scal;
//                         b[i * stride_b + j * ldb + k] = reinterpret_cast<driver_bfloat16&>(product);
//                     }
//                 else
//                     for(size_t k = 0; k < n; ++k)
//                     {   
//                         driver_bfloat16 temp = b[i * stride_b + k * ldb + j];
//                         __nv_bfloat16 product = reinterpret_cast<__nv_bfloat16&>(temp) * scal;
//                         b[i * stride_b + k * ldb + j] = reinterpret_cast<driver_bfloat16&>(product);
//                     }
//             }
//         }
//     }
// }

// clang-format off
// hipblas_initialization string2hipblas_initialization(const std::string& value)
// {
//     return
//         value == "rand_int"   ? hipblas_initialization::rand_int   :
//         value == "trig_float" ? hipblas_initialization::trig_float :
//         value == "hpl"        ? hipblas_initialization::hpl        :
//         static_cast<hipblas_initialization>(-1);
// }
// clang-format on

/* ============================================================================================ */
/*  Convert hipblas constants to lapack char. */

/* ============================================================================================ */
/*! \brief  local handle which is automatically created and destroyed  */

// return precision string for hipblas_datatype
inline constexpr auto hipblas_datatype2string(hipblasDatatype_t type)
{
    switch(type)
    {
    case HIPBLAS_R_16F:
        return "f16_r";
    case HIPBLAS_R_32F:
        return "f32_r";
    case HIPBLAS_R_64F:
        return "f64_r";
    case HIPBLAS_C_16F:
        return "f16_k";
    case HIPBLAS_C_32F:
        return "f32_c";
    case HIPBLAS_C_64F:
        return "f64_c";
    case HIPBLAS_R_8I:
        return "i8_r";
    case HIPBLAS_R_8U:
        return "u8_r";
    case HIPBLAS_R_32I:
        return "i32_r";
    case HIPBLAS_R_32U:
        return "u32_r";
    case HIPBLAS_C_8I:
        return "i8_c";
    case HIPBLAS_C_8U:
        return "u8_c";
    case HIPBLAS_C_32I:
        return "i32_c";
    case HIPBLAS_C_32U:
        return "u32_c";
    case HIPBLAS_R_16B:
        return "bf16_r";
    case HIPBLAS_C_16B:
        return "bf16_c";
    }
    return "invalid";
}

// clang-format off
hipblasDatatype_t string2hipblas_datatype(const std::string& value)
{
    return
        value == "f16_r" || value == "h" ? HIPBLAS_R_16F  :
        value == "f32_r" || value == "s" ? HIPBLAS_R_32F  :
        value == "f64_r" || value == "d" ? HIPBLAS_R_64F  :
        value == "bf16_r"                ? HIPBLAS_R_16B :
        value == "f16_c"                 ? HIPBLAS_C_16B  :
        value == "f32_c" || value == "c" ? HIPBLAS_C_32F  :
        value == "f64_c" || value == "z" ? HIPBLAS_C_64F  :
        value == "bf16_c"                ? HIPBLAS_C_16B :
        value == "i8_r"                  ? HIPBLAS_R_8I   :
        value == "i32_r"                 ? HIPBLAS_R_32I  :
        value == "i8_c"                  ? HIPBLAS_C_8I   :
        value == "i32_c"                 ? HIPBLAS_C_32I  :
        value == "u8_r"                  ? HIPBLAS_R_8U   :
        value == "u32_r"                 ? HIPBLAS_R_32U  :
        value == "u8_c"                  ? HIPBLAS_C_8U   :
        value == "u32_c"                 ? HIPBLAS_C_32U  :
        static_cast<hipblasDatatype_t>(-1);
}

class hipblasLocalHandle
{
    hipblasHandle_t m_handle;
    void*           m_memory = nullptr;

public:
    // hipblasLocalHandle();

    // explicit hipblasLocalHandle(const Arguments& arg);
    hipblasLocalHandle()
    {
        auto status = hipblasCreate(&m_handle);
        if(status != HIPBLAS_STATUS_SUCCESS)
            throw std::runtime_error(hipblasStatusToString(status));
    }

    ~hipblasLocalHandle()
    {
        if(m_memory)
            (hipFree)(m_memory);
        hipblasDestroy(m_handle);
    }

    hipblasLocalHandle(const hipblasLocalHandle&) = delete;
    hipblasLocalHandle(hipblasLocalHandle&&)      = delete;
    hipblasLocalHandle& operator=(const hipblasLocalHandle&) = delete;
    hipblasLocalHandle& operator=(hipblasLocalHandle&&) = delete;

    // Allow hipblasLocalHandle to be used anywhere hipblas_handle is expected
    operator hipblasHandle_t&()
    {
        return m_handle;
    }
    operator const hipblasHandle_t&() const
    {
        return m_handle;
    }
};

char hipblas2char_operation(hipblasOperation_t value)
{
    switch(value)
    {
    case HIPBLAS_OP_N:
        return 'N';
    case HIPBLAS_OP_T:
        return 'T';
    case HIPBLAS_OP_C:
        return 'C';
    }
    return '\0';
}

char hipblas2char_fill(hipblasFillMode_t value)
{
    switch(value)
    {
    case HIPBLAS_FILL_MODE_UPPER:
        return 'U';
    case HIPBLAS_FILL_MODE_LOWER:
        return 'L';
    case HIPBLAS_FILL_MODE_FULL:
        return 'F';
    }
    return '\0';
}

char hipblas2char_diagonal(hipblasDiagType_t value)
{
    switch(value)
    {
    case HIPBLAS_DIAG_UNIT:
        return 'U';
    case HIPBLAS_DIAG_NON_UNIT:
        return 'N';
    }
    return '\0';
}

char hipblas2char_side(hipblasSideMode_t value)
{
    switch(value)
    {
    case HIPBLAS_SIDE_LEFT:
        return 'L';
    case HIPBLAS_SIDE_RIGHT:
        return 'R';
    case HIPBLAS_SIDE_BOTH:
        return 'B';
    }
    return '\0';
}

/* ============================================================================================ */
/*  Convert lapack char constants to hipblas type. */

hipblasOperation_t char2hipblas_operation(char value)
{
    switch(value)
    {
    case 'N':
        return HIPBLAS_OP_N;
    case 'T':
        return HIPBLAS_OP_T;
    case 'C':
        return HIPBLAS_OP_C;
    case 'n':
        return HIPBLAS_OP_N;
    case 't':
        return HIPBLAS_OP_T;
    case 'c':
        return HIPBLAS_OP_C;
    }
    return HIPBLAS_OP_N;
}

hipblasFillMode_t char2hipblas_fill(char value)
{
    switch(value)
    {
    case 'U':
        return HIPBLAS_FILL_MODE_UPPER;
    case 'L':
        return HIPBLAS_FILL_MODE_LOWER;
    case 'u':
        return HIPBLAS_FILL_MODE_UPPER;
    case 'l':
        return HIPBLAS_FILL_MODE_LOWER;
    }
    return HIPBLAS_FILL_MODE_LOWER;
}

hipblasDiagType_t char2hipblas_diagonal(char value)
{
    switch(value)
    {
    case 'U':
        return HIPBLAS_DIAG_UNIT;
    case 'N':
        return HIPBLAS_DIAG_NON_UNIT;
    case 'u':
        return HIPBLAS_DIAG_UNIT;
    case 'n':
        return HIPBLAS_DIAG_NON_UNIT;
    }
    return HIPBLAS_DIAG_NON_UNIT;
}

hipblasSideMode_t char2hipblas_side(char value)
{
    switch(value)
    {
    case 'L':
        return HIPBLAS_SIDE_LEFT;
    case 'R':
        return HIPBLAS_SIDE_RIGHT;
    case 'l':
        return HIPBLAS_SIDE_LEFT;
    case 'r':
        return HIPBLAS_SIDE_RIGHT;
    }
    return HIPBLAS_SIDE_LEFT;
}

// // return precision string for rocblas_datatype
// constexpr auto rocblas_datatype2hipblas_datatype(rocblas_datatype type)
// {
//     switch(type)
//     {
//     case rocblas_datatype_f16_r:
//         return HIPBLAS_R_16F;
//     case rocblas_datatype_f32_r:
//         return HIPBLAS_R_32F;
//     case rocblas_datatype_f64_r:
//         return HIPBLAS_R_64F;
//     case rocblas_datatype_f16_c:
//         return HIPBLAS_C_16F;
//     case rocblas_datatype_f32_c:
//         return HIPBLAS_C_32F;
//     case rocblas_datatype_f64_c:
//         return HIPBLAS_C_64F;
//     case rocblas_datatype_i8_r:
//         return HIPBLAS_R_8I;
//     case rocblas_datatype_u8_r:
//         return HIPBLAS_R_8U;
//     case rocblas_datatype_i32_r:
//         return HIPBLAS_R_32I;
//     case rocblas_datatype_u32_r:
//         return HIPBLAS_R_32U;
//     case rocblas_datatype_i8_c:
//         return HIPBLAS_C_8I;
//     case rocblas_datatype_u8_c:
//         return HIPBLAS_C_8U;
//     case rocblas_datatype_i32_c:
//         return HIPBLAS_C_32I;
//     case rocblas_datatype_u32_c:
//         return HIPBLAS_C_32U;
//     case rocblas_datatype_bf16_r:
//         return HIPBLAS_R_16B;
//     case rocblas_datatype_bf16_c:
//         return HIPBLAS_C_16B;
//     default:
//         return static_cast<hipblasDatatype_t>(-1);
//     }
// }

// constexpr auto rocblas_algo2hipblas_algo(rocblas_gemm_algo_ type)
// {
//     switch(type)
//     {
//     case rocblas_gemm_algo_standard:
//         return HIPBLAS_GEMM_DEFAULT;
//     default:
//         return static_cast<hipblasGemmAlgo_t>(-1);
//     }
// }

#ifndef CHECK_HIPBLAS_ERROR
#define EXPECT_HIPBLAS_STATUS(status, expected)      \
    do                                               \
    {                                                \
        hipblasStatus_t status__ = (status);         \
        if(status__ != expected)                     \
        {                                            \
            fprintf(stderr,                          \
                    "hipBLAS error: %s at %s:%d\n",  \
                    hipblasStatusToString(status__), \
                    __FILE__,                        \
                    __LINE__);                       \
            exit(EXIT_FAILURE);                      \
        }                                            \
    } while(0)
#define CHECK_HIPBLAS_ERROR(STATUS) EXPECT_HIPBLAS_STATUS(STATUS, HIPBLAS_STATUS_SUCCESS)
#endif

template <typename T, bool FORTRAN = false>
hipblasStatus_t hipblasGemm(hipblasHandle_t    handle,
                            hipblasOperation_t transA,
                            hipblasOperation_t transB,
                            int                m,
                            int                n,
                            int                k,
                            const T*           alpha,
                            const T*           A,
                            int                lda,
                            const T*           B,
                            int                ldb,
                            const T*           beta,
                            T*                 C,
                            int                ldc);

template <typename T, bool FORTRAN = false>
hipblasStatus_t hipblasGemmStridedBatched(hipblasHandle_t    handle,
                                          hipblasOperation_t transA,
                                          hipblasOperation_t transB,
                                          int                m,
                                          int                n,
                                          int                k,
                                          const T*           alpha,
                                          const T*           A,
                                          int                lda,
                                          int                bsa,
                                          const T*           B,
                                          int                ldb,
                                          int                bsb,
                                          const T*           beta,
                                          T*                 C,
                                          int                ldc,
                                          int                bsc,
                                          int                batch_count);


// gemm
template <>
hipblasStatus_t hipblasGemm<uint16_t>(hipblasHandle_t    handle,
                                         hipblasOperation_t transA,
                                         hipblasOperation_t transB,
                                         int                m,
                                         int                n,
                                         int                k,
                                         const uint16_t* alpha,
                                         const uint16_t* A,
                                         int                lda,
                                         const uint16_t* B,
                                         int                ldb,
                                         const uint16_t* beta,
                                         uint16_t*       C,
                                         int                ldc)
{
    return hipblasHgemm(handle, transA, transB, m, n, k, reinterpret_cast<const hipblasHalf *>(alpha), reinterpret_cast<const hipblasHalf *>(A), lda, reinterpret_cast<const hipblasHalf *>(B), ldb, reinterpret_cast<const hipblasHalf *>(beta), reinterpret_cast<hipblasHalf *>(C), ldc);
}

template <>
hipblasStatus_t hipblasGemm<float>(hipblasHandle_t    handle,
                                   hipblasOperation_t transA,
                                   hipblasOperation_t transB,
                                   int                m,
                                   int                n,
                                   int                k,
                                   const float*       alpha,
                                   const float*       A,
                                   int                lda,
                                   const float*       B,
                                   int                ldb,
                                   const float*       beta,
                                   float*             C,
                                   int                ldc)
{
    return hipblasSgemm(handle, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

template <>
hipblasStatus_t hipblasGemm<double>(hipblasHandle_t    handle,
                                    hipblasOperation_t transA,
                                    hipblasOperation_t transB,
                                    int                m,
                                    int                n,
                                    int                k,
                                    const double*      alpha,
                                    const double*      A,
                                    int                lda,
                                    const double*      B,
                                    int                ldb,
                                    const double*      beta,
                                    double*            C,
                                    int                ldc)
{
    return hipblasDgemm(handle, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}


// gemm_strided_batched
template <>
hipblasStatus_t hipblasGemmStridedBatched<uint16_t>(hipblasHandle_t    handle,
                                                       hipblasOperation_t transA,
                                                       hipblasOperation_t transB,
                                                       int                m,
                                                       int                n,
                                                       int                k,
                                                       const uint16_t* alpha,
                                                       const uint16_t* A,
                                                       int                lda,
                                                       int                bsa,
                                                       const uint16_t* B,
                                                       int                ldb,
                                                       int                bsb,
                                                       const uint16_t* beta,
                                                       uint16_t*       C,
                                                       int                ldc,
                                                       int                bsc,
                                                       int                batch_count)
{

    return hipblasHgemmStridedBatched(handle,
                                      transA,
                                      transB,
                                      m,
                                      n,
                                      k,
                                      reinterpret_cast<const hipblasHalf *>(alpha),
                                      reinterpret_cast<const hipblasHalf *>(A),
                                      lda,
                                      bsa,
                                      reinterpret_cast<const hipblasHalf *>(B),
                                      ldb,
                                      bsb,
                                      reinterpret_cast<const hipblasHalf *>(beta),
                                      reinterpret_cast<hipblasHalf *>(C),
                                      ldc,
                                      bsc,
                                      batch_count);
}

template <>
hipblasStatus_t hipblasGemmStridedBatched<float>(hipblasHandle_t    handle,
                                                 hipblasOperation_t transA,
                                                 hipblasOperation_t transB,
                                                 int                m,
                                                 int                n,
                                                 int                k,
                                                 const float*       alpha,
                                                 const float*       A,
                                                 int                lda,
                                                 int                bsa,
                                                 const float*       B,
                                                 int                ldb,
                                                 int                bsb,
                                                 const float*       beta,
                                                 float*             C,
                                                 int                ldc,
                                                 int                bsc,
                                                 int                batch_count)
{

    return hipblasSgemmStridedBatched(handle,
                                      transA,
                                      transB,
                                      m,
                                      n,
                                      k,
                                      alpha,
                                      A,
                                      lda,
                                      bsa,
                                      B,
                                      ldb,
                                      bsb,
                                      beta,
                                      C,
                                      ldc,
                                      bsc,
                                      batch_count);
}

template <>
hipblasStatus_t hipblasGemmStridedBatched<double>(hipblasHandle_t    handle,
                                                  hipblasOperation_t transA,
                                                  hipblasOperation_t transB,
                                                  int                m,
                                                  int                n,
                                                  int                k,
                                                  const double*      alpha,
                                                  const double*      A,
                                                  int                lda,
                                                  int                bsa,
                                                  const double*      B,
                                                  int                ldb,
                                                  int                bsb,
                                                  const double*      beta,
                                                  double*            C,
                                                  int                ldc,
                                                  int                bsc,
                                                  int                batch_count)
{

    return hipblasDgemmStridedBatched(handle,
                                      transA,
                                      transB,
                                      m,
                                      n,
                                      k,
                                      alpha,
                                      A,
                                      lda,
                                      bsa,
                                      B,
                                      ldb,
                                      bsb,
                                      beta,
                                      C,
                                      ldc,
                                      bsc,
                                      batch_count);
}

#define driver_gemm hipblasGemm
#define driver_gemm_strided_batched hipblasGemmStridedBatched
#define driver_gemm_ex hipblasGemmEx
// #define driver_algo(X) rocblas_algo2hipblas_algo(X)