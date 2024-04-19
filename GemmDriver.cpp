/* ************************************************************************
 * Copyright (c) <2021> Advanced Micro Devices, Inc.
 *  
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *  
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *  
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 * ************************************************************************ */

#include "utility.hpp"
#include <fstream>
#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef VALIDATE
#include "blis_interface.hpp"
#include "validate.hpp"
#endif

template <typename T>
void BenchGemmStridedBatched(const Arguments& arg, std::promise<std::pair<double,double>> promise)
{
    rocblas_int M = arg.M;
    rocblas_int N = arg.N;
    rocblas_int K = arg.K;

    T h_alpha = arg.get_alpha<T>();
    T h_beta  = arg.get_beta<T>();

    rocblas_int lda = arg.lda;
    rocblas_int ldb = arg.ldb;
    rocblas_int ldc = arg.ldc;

    rocblas_stride stride_a    = arg.stride_a;
    rocblas_stride stride_b    = arg.stride_b;
    rocblas_stride stride_c    = arg.stride_c;
    rocblas_int batch_count = arg.batch_count;

    rocblas_operation transA = char2rocblas_operation(arg.transA);
    rocblas_operation transB = char2rocblas_operation(arg.transB);

    rocblas_local_handle handle;

    rocblas_int A_row = transA == rocblas_operation_none ? M : K;
    rocblas_int A_col = transA == rocblas_operation_none ? K : M;
    rocblas_int B_row = transB == rocblas_operation_none ? K : N;
    rocblas_int B_col = transB == rocblas_operation_none ? N : K;

    // Early exit
    if(!M || !N || !batch_count)
        return;

    // check here to prevent undefined memory allocation error
    if(M < 0 || N < 0 || K < 0 || lda < A_row || ldb < B_row || ldc < M || batch_count < 0)
    {
        rocblas_cout << "Invalid sizes...exiting" << std::endl;
        exit(1);
    }

    rocblas_int reinit_c = arg.reinit_c && h_beta != 0;
    rocblas_int time_each_iter = arg.time_each_iter || reinit_c;
    double      host_time;
    double      rocblas_gflops;
    static double cpu_time_used, cblas_gflops;
    int         deviceId;
    if(multi_device>1)
        CHECK_HIP_ERROR(hipGetDevice(&deviceId));

    double rocblas_error = 0.0;

    bool vChecks = (arg.unit_check || arg.norm_check);
    bool transferOutput = (vChecks || storeOutputData);

    static host_strided_batch_matrix<T> hA(A_row, A_col, lda, stride_a, batch_count);
    static host_strided_batch_matrix<T> hB(B_row, B_col, ldb, stride_b, batch_count);
    host_strided_batch_matrix<T> hC(M, N, ldc, stride_c, batch_count);

    static host_strided_batch_matrix<T> hC_orig = (arg.reinit_c) ? host_strided_batch_matrix<T>(M, N, ldc, stride_c, batch_count)
                           : host_strided_batch_matrix<T>(0, 1, 1, 1, 1);
    static host_strided_batch_matrix<T> hC_gold = (vChecks) ? host_strided_batch_matrix<T>(M, N, ldc, stride_c, batch_count)
                           : host_strided_batch_matrix<T>(0, 1, 1, 1, 1);

    // Check host memory allocation
    CHECK_HIP_ERROR(hA.memcheck());
    CHECK_HIP_ERROR(hB.memcheck());
    CHECK_HIP_ERROR(hC.memcheck());
    CHECK_HIP_ERROR(hC_gold.memcheck());

    // allocate memory on device
    device_strided_batch_matrix<T> dA(A_row, A_col, lda, stride_a, batch_count);
    device_strided_batch_matrix<T> dB(B_row, B_col, ldb, stride_b, batch_count);
    device_strided_batch_matrix<T> dC(M, N, ldc, stride_c, batch_count);

    device_vector<T> d_alpha(1);
    device_vector<T> d_beta(1);
    
    // Check device memory allocation
    CHECK_HIP_ERROR(dA.memcheck());
    CHECK_HIP_ERROR(dB.memcheck());
    CHECK_HIP_ERROR(dC.memcheck());
    CHECK_HIP_ERROR(d_alpha.memcheck());
    CHECK_HIP_ERROR(d_beta.memcheck());

    // Initial Data on CPU
    if((multi_device>1 && deviceId==0) || multi_device == 1)
    {
        if(arg.initialization == rocblas_initialization_file)
            loadFromBin<T>(hA, a_file, hB, b_file, hC, c_file);
        else
        {
            // Initialize data on host memory
            rocblas_init_matrix(
                hA, arg, rocblas_client_alpha_sets_nan, rocblas_client_general_matrix, true);
            rocblas_init_matrix<T, true>(
                hB, arg, rocblas_client_alpha_sets_nan, rocblas_client_general_matrix, false, true);
            rocblas_init_matrix<T, true>(hC, arg, rocblas_client_beta_sets_nan, rocblas_client_general_matrix);

            if(arg.initialization == rocblas_initialization_random_broad)
                normalizeInputs<T>(hA,hB);
        }

        if(vChecks)
            hC_gold.copy_from(hC);
        if(reinit_c)
            hC_orig.copy_from(hC);
        memBarrier.wait();
    }
    else
        memBarrier.wait();

    if(storeInitData)
    {
        storeInitToBin<T,T>(hA, a_file, hB, b_file, hC, c_file);
    }

    // copy data from CPU to device
    CHECK_HIP_ERROR(dA.transfer_from(hA));
    CHECK_HIP_ERROR(dB.transfer_from(hB));
    CHECK_HIP_ERROR(dC.transfer_from(hC));

#ifdef VALIDATE
    if(arg.norm_check)
    {
        // ROCBLAS rocblas_pointer_mode_host
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

        CHECK_ROCBLAS_ERROR(rocblas_gemm_strided_batched<T>(handle,
                                                            transA,
                                                            transB,
                                                            M,
                                                            N,
                                                            K,
                                                            &h_alpha,
                                                            dA,
                                                            lda,
                                                            stride_a,
                                                            dB,
                                                            ldb,
                                                            stride_b,
                                                            &h_beta,
                                                            dC,
                                                            ldc,
                                                            stride_c,
                                                            batch_count));
        CHECK_HIP_ERROR(hC.transfer_from(dC));

        // ROCBLAS rocblas_pointer_mode_device
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));

        CHECK_HIP_ERROR(dC.transfer_from(hC_gold));
        CHECK_HIP_ERROR(hipMemcpy(d_alpha, &h_alpha, sizeof(T), hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(d_beta, &h_beta, sizeof(T), hipMemcpyHostToDevice));

        CHECK_ROCBLAS_ERROR(rocblas_gemm_strided_batched<T>(handle,
                                                            transA,
                                                            transB,
                                                            M,
                                                            N,
                                                            K,
                                                            d_alpha,
                                                            dA,
                                                            lda,
                                                            stride_a,
                                                            dB,
                                                            ldb,
                                                            stride_b,
                                                            d_beta,
                                                            dC,
                                                            ldc,
                                                            stride_c,
                                                            batch_count));

        if(multi_device > 1 && deviceId!=0)
        {
            memBarrier2.wait(deviceId);
        }

        if(multi_device==1 || (multi_device > 1 && deviceId==0))
        {
            // CPU BLAS
            cpu_time_used = get_time_us();
            for(rocblas_int i = 0; i < batch_count; i++)
            {
                blis_gemm<T>(transA,
                            transB,
                            M,
                            N,
                            K,
                            h_alpha,
                            hA + stride_a * i,
                            lda,
                            hB + stride_b * i,
                            ldb,
                            h_beta,
                            hC_gold + stride_c * i,
                            ldc);
            }
            cpu_time_used = get_time_us() - cpu_time_used;
            cblas_gflops  = gemm_gflop_count<T>(M, N, K) * batch_count / cpu_time_used * 1e6;

            if(multi_device > 1)
            {
                memBarrier2.wait(deviceId);
            }
        }

        for(int i=0; i<2; i++)
        {
            if(arg.unit_check)
            {
                if(std::is_same_v<T, rocblas_half> && K > 10000)
                {
                    // For large K, rocblas_half tends to diverge proportional to K
                    // Tolerance is slightly greater than 1 / 1024.0
                    const double tol = K * sum_error_tolerance<T>;
                    near_check_general<T>(M, N, batch_count, ldc, stride_c, hC_gold, hC, tol);
                }
                else
                {
                    unit_check_general<T>(M, N, batch_count, ldc, stride_c, hC_gold, hC);
                }
            }

            if(arg.norm_check)
            {
                double error
                    = fabs(norm_check_general<T>('F', M, N, ldc, stride_c, batch_count, hC_gold, hC));

                rocblas_error = error > rocblas_error ? error : rocblas_error;
            }
            if(i==0)
            {
                CHECK_HIP_ERROR(hC.transfer_from(dC));
            }
        }
    }
#endif

    int number_cold_calls = 2;
    int number_hot_calls  = arg.iters;
    hipEvent_t start, stop, flush;
    CHECK_HIP_ERROR(hipEventCreateWithFlags(&flush, hipEventReleaseToSystem));
    CHECK_HIP_ERROR(hipEventCreate(&start));
    CHECK_HIP_ERROR(hipEventCreate(&stop));
    float kernel_time = 0.0f;
    host_time        = 0.0;
    float kernel_time_iter = 0.0f;
    double host_time_iter = 0.0f;

    CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

    for(int i = 0; i < number_cold_calls; i++)
    {
        rocblas_gemm_strided_batched<T>(handle,
                                        transA,
                                        transB,
                                        M,
                                        N,
                                        K,
                                        &h_alpha,
                                        dA,
                                        lda,
                                        stride_a,
                                        dB,
                                        ldb,
                                        stride_b,
                                        &h_beta,
                                        dC,
                                        ldc,
                                        stride_c,
                                        batch_count);
    }

    if(time_each_iter)
    {
        for(int i = 0; i < number_hot_calls; i++)
        {
            if(reinit_c && ((arg.norm_check && i == 0) || i > 0))
                CHECK_HIP_ERROR(dC.transfer_from(hC_orig));
            if(arg.flush_gpu_cache)
                CHECK_HIP_ERROR(hipEventRecord(flush, NULL));

            host_time_iter = get_time_us();
            CHECK_HIP_ERROR(hipEventRecord(start, NULL));

            rocblas_gemm_strided_batched<T>(handle,
                                        transA,
                                        transB,
                                        M,
                                        N,
                                        K,
                                        &h_alpha,
                                        dA,
                                        lda,
                                        stride_a,
                                        dB,
                                        ldb,
                                        stride_b,
                                        &h_beta,
                                        dC,
                                        ldc,
                                        stride_c,
                                        batch_count);

            CHECK_HIP_ERROR(hipEventRecord(stop, NULL));
            CHECK_HIP_ERROR(hipEventSynchronize(stop));
            host_time += get_time_us() - host_time_iter;
            CHECK_HIP_ERROR(hipEventElapsedTime(&kernel_time_iter, start, stop));
            kernel_time+=kernel_time_iter;
        }
    }
    else
    {
        std::pair<double,double> times;
        if(multi_device>1)
        {
            usleep(0.5 * 1000000);
            perfBarrier.wait(deviceId);
        }
        times.first = get_time_us(); // in microseconds
        CHECK_HIP_ERROR(hipEventRecord(start, NULL));
        for(int i = 0; i < number_hot_calls; i++)
        {
            rocblas_gemm_strided_batched<T>(handle,
                                        transA,
                                        transB,
                                        M,
                                        N,
                                        K,
                                        &h_alpha,
                                        dA,
                                        lda,
                                        stride_a,
                                        dB,
                                        ldb,
                                        stride_b,
                                        &h_beta,
                                        dC,
                                        ldc,
                                        stride_c,
                                        batch_count);
        }

        CHECK_HIP_ERROR(hipEventRecord(stop, NULL));
        CHECK_HIP_ERROR(hipEventSynchronize(stop));
        times.second = get_time_us();
        if(multi_device>1)
            promise.set_value(times);
        CHECK_HIP_ERROR(hipEventElapsedTime(&kernel_time, start, stop));
        host_time = times.second-times.first;
    }

    if(storeOutputData)
    {
        CHECK_HIP_ERROR(hC.transfer_from(dC));
        storeOutputToBin<T>(hC, o_file);
    }

    rocblas_gflops = gemm_gflop_count<T>(M, N, K) * batch_count * number_hot_calls  / kernel_time * 1e3;

    std::stringstream msg;
    if(multi_device>1)
    {
        double host_gflops = gemm_gflop_count<T>(M, N, K) * number_hot_calls / (host_time) * 1e6;
        msg << "Device " << deviceId << std::endl
        << "transA,transB,M,N,K,alpha,lda,stride_a,ldb,stride_b,beta,ldc,stride_c,Batch_"
            "Count,rocblas-Gflops,rocblas-Gflops(using host_time),host_time(us),kernel_time(us)" << std::endl
        << arg.transA << "," << arg.transB << "," << M << "," << N << "," << K << "," << arg.get_alpha<T>() 
        << "," << lda << "," << stride_a << "," << ldb << "," << stride_b << "," << arg.get_beta<T>() 
        << "," << ldc << "," << stride_c << "," << batch_count << "," << rocblas_gflops << "," << host_gflops << "," 
        << host_time / number_hot_calls << "," << kernel_time/number_hot_calls*1000 << std::endl;
    }
    else
        msg << "transA,transB,M,N,K,alpha,lda,stride_a,ldb,stride_b,beta,ldc,stride_c,Batch_"
            "Count,rocblas-Gflops,host_time(us),kernel_time(us)" << std::endl
        << arg.transA << "," << arg.transB << "," << M << "," << N << "," << K << "," << arg.get_alpha<T>() 
        << "," << lda << "," << stride_a << "," << ldb << "," << stride_b << "," << arg.get_beta<T>() 
        << "," << ldc << "," << stride_c << "," << batch_count << "," << rocblas_gflops << "," 
        << host_time / number_hot_calls << "," << kernel_time/number_hot_calls*1000 << std::endl;

    if(arg.norm_check)
    {
        msg << "cblas-Gflops,us,rocblas-error" << std::endl
        << cblas_gflops << "," << cpu_time_used << "," << rocblas_error << std::endl;
    }

    rocblas_cout << msg.str();
}

template <typename Ti, typename To, typename Tc>
void BenchGemmEx(Arguments& arg, std::promise<std::pair<double,double>> promise)
{
    rocblas_gemm_algo algo           = static_cast<rocblas_gemm_algo>(arg.algo);
    int32_t           solution_index = arg.solution_index;
    uint32_t          flags          = arg.flags;

    bool nantest = rocblas_isnan(arg.beta) || rocblas_isnan(arg.betai);
    if(!std::is_same_v<To, float> && !std::is_same<To, double>{}
       && !std::is_same_v<To, rocblas_half> && !rocblas_is_complex<To> && nantest)
        return; // Exclude integers or other types which don't support NaN

    Tc h_alpha_Tc = arg.get_alpha<Tc>();
    Tc h_beta_Tc  = arg.get_beta<Tc>();

    rocblas_int c_equals_d = arg.c_equals_d;
    rocblas_int reinit_c = arg.reinit_c && h_beta_Tc != 0 && c_equals_d;
    rocblas_int time_each_iter = arg.time_each_iter || reinit_c || arg.flush_gpu_cache;
    rocblas_int tensile_timing = arg.tensile_timing;

    double        host_time;
    double        rocblas_gflops;
    static double cpu_time_used, cblas_gflops;
    double        rocblas_error = 0.0;
    int           deviceId;

    if(multi_device>1)
        CHECK_HIP_ERROR(hipGetDevice(&deviceId));

    rocblas_local_handle handle;
    auto                 transA = char2rocblas_operation(arg.transA);
    auto                 transB = char2rocblas_operation(arg.transB);
    auto                 M = arg.M, N = arg.N, K = arg.K;
    auto                 lda = arg.lda, ldb = arg.ldb, ldc = arg.ldc, ldd = arg.ldd;
    auto                 A_row = transA == rocblas_operation_none ? M : K;
    auto                 A_col = transA == rocblas_operation_none ? K : M;
    auto                 B_row = transB == rocblas_operation_none ? K : N;
    auto                 B_col = transB == rocblas_operation_none ? N : K;
    auto                 d_type = arg.d_type;

    // check for invalid sizes
    if(M < 0 || N < 0 || K < 0 || lda < A_row || ldb < B_row || ldc < M || (ldd < M && !c_equals_d)
       || (std::is_same_v<Ti, int8_t> 
           && (K % 4 != 0 || (transA != rocblas_operation_none && lda % 4 != 0)
               || (transB == rocblas_operation_none && ldb % 4 != 0))))
    {
        rocblas_cout << "Invalid sizes...exiting" << std::endl;
        exit(1);
    }

    if(arg.c_equals_d)
    {
        ldd    = ldc;
        d_type = arg.c_type;
    }

    rocblas_stride stride_a = size_t(lda) * A_col;
    rocblas_stride stride_b = size_t(ldb) * B_col;
    rocblas_stride stride_c = size_t(ldc) * N;
    rocblas_stride stride_d = !arg.c_equals_d ? size_t(ldd) * N : 0;

    rocblas_stride aligned_stride_a = align_stride<Ti>(stride_a);
    rocblas_stride aligned_stride_b = align_stride<Ti>(stride_b);
    rocblas_stride aligned_stride_c = align_stride<To>(stride_c);
    rocblas_stride aligned_stride_d = align_stride<To>(stride_d);

    // size_t flush_batch_count = 1;
    // if(arg.timing)
    // {
        // size_t a_size = M * K * sizeof(Ti);
        // size_t b_size = K * N * sizeof(Ti);
        // size_t c_size = M * N * sizeof(To);
        // //      exclude d_size from cached_size calculation because
        // //      - for arg.outofplace == false : D == C
        // //      - for arg.outofplace == true  : D is write only
        // size_t a_b_c_cached_size = a_size + b_size + c_size;

        // flush_batch_count = calculate_flush_batch_count(
        //     arg.flush_batch_count, arg.flush_memory_size, a_b_c_cached_size);
    // }

    bool vChecks = (arg.unit_check || arg.norm_check);
    bool transferOutput = (vChecks || storeOutputData);

    // Allocate host memory
    static host_matrix<Ti> hA(A_row, A_col, lda);
    static host_matrix<Ti> hB(B_row, B_col, ldb);
    static host_matrix<To> hC(M, N, ldc);

    host_matrix<To> hD      =  transferOutput ? host_matrix<To>(M, N, ldd) : host_matrix<To>(0, 1, 1);
    static host_matrix<To> hD_gold =  vChecks ? host_matrix<To>(M, N, ldd) : host_matrix<To>(0, 1, 1);

    // Check host memory allocation
    CHECK_HIP_ERROR(hA.memcheck());
    CHECK_HIP_ERROR(hB.memcheck());
    CHECK_HIP_ERROR(hC.memcheck());

    // allocate memory on device
    device_strided_batch_matrix<Ti> dA(A_row, A_col, lda, aligned_stride_a, 1);
    device_strided_batch_matrix<Ti> dB(B_row, B_col, ldb, aligned_stride_b, 1);
    device_strided_batch_matrix<To> dC(M, N, ldc, aligned_stride_c, 1);
    // if C!=D, allocate C and D normally
    // if C==D, allocate C big enough for the larger of C and D; D points to C
    device_strided_batch_matrix<To> dD_alloc
        = !(arg.c_equals_d) ? device_strided_batch_matrix<To>(M, N, ldd, aligned_stride_d, 1)
                           : device_strided_batch_matrix<To>(0, 1, 1, 1, 1);
    device_strided_batch_matrix<To>& dD = !(arg.c_equals_d) ? dD_alloc : dC;

    device_vector<Tc> d_alpha_Tc(1);
    device_vector<Tc> d_beta_Tc(1);

    // Check device memory allocation
    CHECK_HIP_ERROR(dA.memcheck());
    CHECK_HIP_ERROR(dB.memcheck());
    CHECK_HIP_ERROR(dC.memcheck());
    CHECK_HIP_ERROR(dD_alloc.memcheck());
    CHECK_HIP_ERROR(d_alpha_Tc.memcheck());
    CHECK_HIP_ERROR(d_beta_Tc.memcheck());

    if((multi_device>1 && deviceId==0) || multi_device == 1)
    {
        if(arg.initialization == rocblas_initialization_file)
            loadFromBin<Ti, To>(hA, a_file, hB, b_file, hC, c_file);
        else
        {
            // Initialize data on host memory
            rocblas_init_matrix(
                hA, arg, rocblas_client_alpha_sets_nan, rocblas_client_general_matrix, true);
            rocblas_init_matrix<Ti, true>(
                hB, arg, rocblas_client_alpha_sets_nan, rocblas_client_general_matrix, false, true);
            rocblas_init_matrix<To, true>(
                hC, arg, rocblas_client_beta_sets_nan, rocblas_client_general_matrix);

            if(arg.initialization == rocblas_initialization_random_broad)
                normalizeInputs<Ti>(hA,hB);
        }

        if(vChecks)
        {
            rocblas_init_nan<To>(hD, M, N, ldd);
            rocblas_init_nan<To>(hD_gold, M, N, ldd);
        }
             
        memBarrier.wait();
    }
    else
        memBarrier.wait();

    if(storeInitData)
    {
        storeInitToBin<Ti,To>(hA, a_file, hB, b_file, hC, c_file);
    }

    // copy data from CPU to device

    // copy data from CPU to device
    CHECK_HIP_ERROR(dA.broadcast_one_matrix_from(hA));
    CHECK_HIP_ERROR(dB.broadcast_one_matrix_from(hB));
    CHECK_HIP_ERROR(dC.broadcast_one_matrix_from(hC));

#ifdef VALIDATE
    if(arg.norm_check || arg.unit_check)
    {
        // ROCBLAS rocblas_pointer_mode_host
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

        CHECK_ROCBLAS_ERROR(rocblas_gemm_ex(handle,
                                            transA,
                                            transB,
                                            M,
                                            N,
                                            K,
                                            &h_alpha_Tc,
                                            dA[0],
                                            arg.a_type,
                                            lda,
                                            dB[0],
                                            arg.b_type,
                                            ldb,
                                            &h_beta_Tc,
                                            dC[0],
                                            arg.c_type,
                                            ldc,
                                            dD[0],
                                            d_type,
                                            ldd,
                                            arg.compute_type,
                                            algo,
                                            solution_index,
                                            flags));                     

        CHECK_HIP_ERROR(hD.transfer_one_matrix_from(dD));

        // ROCBLAS rocblas_pointer_mode_device
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
        CHECK_HIP_ERROR(hipMemcpy(d_alpha_Tc, &h_alpha_Tc, sizeof(Tc), hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(d_beta_Tc, &h_beta_Tc, sizeof(Tc), hipMemcpyHostToDevice));

        CHECK_HIP_ERROR(dC.broadcast_one_matrix_from(hC));

        CHECK_ROCBLAS_ERROR(rocblas_gemm_ex(handle,
                                            transA,
                                            transB,
                                            M,
                                            N,
                                            K,
                                            d_alpha_Tc,
                                            dA[0],
                                            arg.a_type,
                                            lda,
                                            dB[0],
                                            arg.b_type,
                                            ldb,
                                            d_beta_Tc,
                                            dC[0],
                                            arg.c_type,
                                            ldc,
                                            dD[0],
                                            d_type,
                                            ldd,
                                            arg.compute_type,
                                            algo,
                                            solution_index,
                                            flags));

        if(multi_device > 1 && deviceId!=0)
        {
            memBarrier2.wait(deviceId);
        }

        if(multi_device==1 || (multi_device > 1 && deviceId==0))
        {
            // CPU BLAS
            // copy C matrix into D matrix
            copy_matrix_with_different_leading_dimensions(hC, hD_gold);
            cpu_time_used = get_time_us();
            blis_gemm<Ti,To,Tc>(
                transA, transB, M, N, K, h_alpha_Tc, hA, lda, hB, ldb, h_beta_Tc, hD_gold, ldd);
            //if C does not equal D check if C changed

            cpu_time_used = get_time_us() - cpu_time_used;
            cblas_gflops  = gemm_gflop_count<To>(M, N, K) / cpu_time_used * 1e6;

            if(multi_device > 1)
            {
                memBarrier2.wait(deviceId);
            }
        }

        //releasing already used host memory
        hA=host_matrix<Ti>();
        hB=host_matrix<Ti>();
        if(!reinit_c)
            hC=host_matrix<To>();

        for(int i=0; i<2; i++)
        {
            if(arg.unit_check)
            {
                if(std::is_same_v<Tc, rocblas_half> && K > 10000)
                {
                    // For large K, rocblas_half tends to diverge proportional to K
                    // Tolerance is slightly greater than 1 / 1024.0
                    const double tol = K * sum_error_tolerance<Tc>;
                    near_check_general<To>(M, N, ldd, (To*)hD_gold, (To*)hD, tol);
                }
                else
                {
                    unit_check_general<To>(M, N, ldd, (To*)hD_gold, (To*)hD);
                }
            }

            if(arg.norm_check)
            {
                auto err = 0.0;
                err = fabs(norm_check_general<To>('F', M, N, ldd, (To*)hD_gold, (To*)hD));

                rocblas_error = err > rocblas_error ? err : rocblas_error;
            }

            if(i==0)
            {
                CHECK_HIP_ERROR(hD.transfer_one_matrix_from(dD));
            }
        }
    }
#endif

    int number_cold_calls = 2;
    int number_hot_calls  = arg.iters;
    int numEvents = (tensile_timing ? number_hot_calls + 1: 1);

    hipEvent_t flush, start[numEvents], stop[numEvents];
    CHECK_HIP_ERROR(hipEventCreateWithFlags(&flush, hipEventReleaseToSystem));

    for(int i =0; i < numEvents;i++)
    {
        CHECK_HIP_ERROR(hipEventCreate(&start[i]));
        CHECK_HIP_ERROR(hipEventCreate(&stop[i]));
    }

    float kernel_time = 0.0f;
    float tensile_time = 0.0f;
    host_time        = 0.0;
    float kernel_time_iter = 0.0f;
    double host_time_iter = 0.0f;

    CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

    for(int i = 0; i < number_cold_calls; i++)
    {
        CHECK_ROCBLAS_ERROR(rocblas_gemm_ex(handle,
                                            transA,
                                            transB,
                                            M,
                                            N,
                                            K,
                                            &h_alpha_Tc,
                                            dA[0],
                                            arg.a_type,
                                            lda,
                                            dB[0],
                                            arg.b_type,
                                            ldb,
                                            &h_beta_Tc,
                                            dC[0],
                                            arg.c_type,
                                            ldc,
                                            dD[0],
                                            d_type,
                                            ldd,
                                            arg.compute_type,
                                            algo,
                                            solution_index,
                                            flags));
    }

    if(time_each_iter)
    {
        for(int i = 0; i < number_hot_calls; i++)
        {
            if(reinit_c && ((arg.norm_check && i == 0) || i > 0))
                CHECK_HIP_ERROR(dC.broadcast_one_matrix_from(hC));
            if(arg.flush_gpu_cache)
                CHECK_HIP_ERROR(hipEventRecord(flush, NULL));

            host_time_iter = get_time_us();
            CHECK_HIP_ERROR(hipEventRecord(start[numEvents-1], NULL));

            rocblas_gemm_ex(handle,
                            transA,
                            transB,
                            M,
                            N,
                            K,
                            &h_alpha_Tc,
                            dA[0],
                            arg.a_type,
                            lda,
                            dB[0],
                            arg.b_type,
                            ldb,
                            &h_beta_Tc,
                            dC[0],
                            arg.c_type,
                            ldc,
                            dD[0],
                            d_type,
                            ldd,
                            arg.compute_type,
                            algo,
                            solution_index,
                            flags);

            CHECK_HIP_ERROR(hipEventRecord(stop[numEvents-1], NULL));
            CHECK_HIP_ERROR(hipEventSynchronize(stop[numEvents-1]));
            host_time += get_time_us() - host_time_iter;
            CHECK_HIP_ERROR(hipEventElapsedTime(&kernel_time_iter, start[numEvents-1], stop[numEvents-1]));
            kernel_time+=kernel_time_iter;
        }
    }
    else
    {
        std::pair<double,double> times;
        if(multi_device>1)
        {
            usleep(0.5 * 1000000);
            perfBarrier.wait(deviceId);
        }
        times.first = get_time_us(); // in microseconds
        CHECK_HIP_ERROR(hipEventRecord(start[numEvents-1], NULL));
        for(int i = 0; i < number_hot_calls; i++)
        {
            ROCBLAS_INVOKE_START_STOP_EVENTS(handle, tensile_timing ? start[i]: nullptr, tensile_timing ? stop[i] : nullptr,
            rocblas_gemm_ex(handle,
                            transA,
                            transB,
                            M,
                            N,
                            K,
                            &h_alpha_Tc,
                            dA[0],
                            arg.a_type,
                            lda,
                            dB[0],
                            arg.b_type,
                            ldb,
                            &h_beta_Tc,
                            dC[0],
                            arg.c_type,
                            ldc,
                            dD[0],
                            d_type,
                            ldd,
                            arg.compute_type,
                            algo,
                            solution_index,
                            flags));
        }

        CHECK_HIP_ERROR(hipEventRecord(stop[numEvents-1], NULL));
        CHECK_HIP_ERROR(hipEventSynchronize(stop[numEvents-1]));

        times.second = get_time_us();
        if(multi_device>1)
            promise.set_value(times);
        host_time = times.second-times.first;
        for(int i=0; i<numEvents-1;i++)
        {
            CHECK_HIP_ERROR(hipEventElapsedTime(&kernel_time_iter, start[i], stop[i]));
            tensile_time+=kernel_time_iter;
        }

        CHECK_HIP_ERROR(hipEventElapsedTime(&kernel_time, start[numEvents-1], stop[numEvents-1]));
    }

    if(storeOutputData)
    {
        CHECK_HIP_ERROR(hD.transfer_one_matrix_from(dD));
        storeOutputToBin<To>(hD, o_file);
    }

    rocblas_gflops = gemm_gflop_count<Ti>(M, N, K) * number_hot_calls / (tensile_timing ? tensile_time : kernel_time) * 1e3;
    double host_gflops = gemm_gflop_count<Ti>(M, N, K) * number_hot_calls / (host_time) * 1e6;

    std::stringstream msg;
    if(tensile_timing)
    {
        if(multi_device>1)
        {
            msg << "Device " << deviceId << std::endl
            << "transA,transB,M,N,K,alpha,lda,ldb,beta,ldc,rocblas-Gflops,rocblas-Gflops(using host_time),host_time(us),kernel_time(us)" 
            << ",tensile_time(us)" << std::endl << rocblas2char_operation(transA) << "," << rocblas2char_operation(transB) << ","
            << M << "," << N << "," << K << "," << arg.alpha << "," << lda << "," << ldb
            << "," << arg.beta << "," << ldc << "," << rocblas_gflops << "," << host_gflops << ","
            << host_time / number_hot_calls << "," << kernel_time/number_hot_calls*1000 << ","
            << tensile_time/number_hot_calls*1000 << std::endl;
        }
        else
            msg << "transA,transB,M,N,K,alpha,lda,ldb,beta,ldc,rocblas-Gflops,host_time(us),kernel_time(us)" 
            << ",tensile_time(us)" << std::endl << rocblas2char_operation(transA) << "," << rocblas2char_operation(transB) << ","
            << M << "," << N << "," << K << "," << arg.alpha << "," << lda << "," << ldb
            << "," << arg.beta << "," << ldc << "," << rocblas_gflops << ","
            << host_time / number_hot_calls << "," << kernel_time/number_hot_calls*1000 << ","
            << tensile_time/number_hot_calls*1000 << std::endl;    
    }
    else
    {
        if(multi_device>1)
        {
            msg << "Device " << deviceId << std::endl
            << "transA,transB,M,N,K,alpha,lda,ldb,beta,ldc,rocblas-Gflops,rocblas-Gflops(using host_time),host_time(us),kernel_time(us)"<< std::endl
            << rocblas2char_operation(transA) << "," << rocblas2char_operation(transB) << ","
            << M << "," << N << "," << K << "," << arg.alpha << "," << lda << "," << ldb
            << "," << arg.beta << "," << ldc << "," << rocblas_gflops << "," << host_gflops << ","
            << host_time / number_hot_calls << "," << kernel_time/number_hot_calls*1000 << std::endl;
        }
        else
            msg << "transA,transB,M,N,K,alpha,lda,ldb,beta,ldc,rocblas-Gflops,host_time(us),kernel_time(us)"<< std::endl
                << rocblas2char_operation(transA) << "," << rocblas2char_operation(transB) << ","
            << M << "," << N << "," << K << "," << arg.alpha << "," << lda << "," << ldb
            << "," << arg.beta << "," << ldc << "," << rocblas_gflops  << ","
            << host_time / number_hot_calls << "," << kernel_time/number_hot_calls*1000 << std::endl;
    }

    if(arg.norm_check)
    {
        msg << "cblas-Gflops,us,rocblas-error" << std::endl
        << cblas_gflops << "," << cpu_time_used << "," << rocblas_error << std::endl;
    }

    rocblas_cout << msg.str();
}

template <typename T>
void BenchGemm(Arguments& arg, std::promise<std::pair<double,double>> promise)
{
    rocblas_operation transA = char2rocblas_operation(arg.transA);
    rocblas_operation transB = char2rocblas_operation(arg.transB);

    rocblas_int M = arg.M;
    rocblas_int N = arg.N;
    rocblas_int K = arg.K;

    rocblas_int lda = arg.lda;
    rocblas_int ldb = arg.ldb;
    rocblas_int ldc = arg.ldc;

    T h_alpha = arg.get_alpha<T>();
    T h_beta  = arg.get_beta<T>();

    rocblas_int          reinit_c = arg.reinit_c && h_beta != 0;
    rocblas_int          time_each_iter = arg.time_each_iter || reinit_c;
    double               host_time;
    double               rocblas_gflops;
    static double        cblas_gflops, cpu_time_used;
    double               rocblas_error = 0.0;
    rocblas_local_handle handle;
    int deviceId;
    if(multi_device>1)
        CHECK_HIP_ERROR(hipGetDevice(&deviceId));

    rocblas_int A_row = transA == rocblas_operation_none ? M : K;
    rocblas_int A_col = transA == rocblas_operation_none ? K : M;
    rocblas_int B_row = transB == rocblas_operation_none ? K : N;
    rocblas_int B_col = transB == rocblas_operation_none ? N : K;

    // check here to prevent undefined memory allocation error
    if(M < 0 || N < 0 || K < 0 || lda < A_row || ldb < B_row || ldc < M)
    {
        rocblas_cout << "Invalid sizes...exiting" << std::endl;
        exit(1);
    }

    // allocate memory on device
    device_matrix<T> dA(A_row, A_col, lda);
    device_matrix<T> dB(B_row, B_col, ldb);
    device_matrix<T> dC(M, N, ldc);
    device_vector<T> d_alpha(1, 1);
    device_vector<T> d_beta(1, 1);

    // Check device memory allocation
    CHECK_HIP_ERROR(dA.memcheck());
    CHECK_HIP_ERROR(dB.memcheck());
    CHECK_HIP_ERROR(dC.memcheck());
    CHECK_HIP_ERROR(d_alpha.memcheck());
    CHECK_HIP_ERROR(d_beta.memcheck());

    bool vChecks = (arg.unit_check || arg.norm_check);
    bool transferOutput = (vChecks || storeOutputData);

    // Naming: dX is in GPU (device) memory. hK is in CPU (host) memory
    static host_matrix<T> hA(A_row, A_col, lda);
    static host_matrix<T> hB(B_row, B_col, ldb);
    host_matrix<T> hC(M, N, ldc);
    static host_matrix<T> hC_gold = (vChecks) ? host_matrix<T>(M, N, ldc)
                            : host_matrix<T>(0, 0, 0);
    static host_matrix<T> hC_orig = (arg.reinit_c) ? host_matrix<T>(M, N, ldc)
                            : host_matrix<T>(0, 0, 0);

    // Initial Data on CPU
    if((multi_device>1 && deviceId==0) || multi_device == 1)
    {

        if(arg.initialization == rocblas_initialization_file)
            loadFromBin<T>(hA, a_file, hB, b_file, hC, c_file);
        else
        {
            // Initialize data on host memory
            rocblas_init_matrix(
                hA, arg, rocblas_client_alpha_sets_nan, rocblas_client_general_matrix, true);
            rocblas_init_matrix<T, true>(
                hB, arg, rocblas_client_alpha_sets_nan, rocblas_client_general_matrix, false, true);
            rocblas_init_matrix<T, true>(hC, arg, rocblas_client_beta_sets_nan, rocblas_client_general_matrix);

            if(arg.initialization == rocblas_initialization_random_broad)
                normalizeInputs<T>(hA,hB);
        }

        if(reinit_c)
            hC_orig = hC;

        if(vChecks)
            hC_gold = hC;
        memBarrier.wait();
    }
    else
    {
        memBarrier.wait();
    }

    if(storeInitData)
    {
        storeInitToBin<T,T>(hA, a_file, hB, b_file, hC, c_file);
    }

    // copy data from CPU to device
    CHECK_HIP_ERROR(dA.transfer_from(hA));
    CHECK_HIP_ERROR(dB.transfer_from(hB));
    CHECK_HIP_ERROR(dC.transfer_from(hC));

#ifdef VALIDATE
    if(arg.norm_check || arg.unit_check)
    {
        // ROCBLAS rocblas_pointer_mode_host
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
        CHECK_ROCBLAS_ERROR(rocblas_gemm<T>(
            handle, transA, transB, M, N, K, &h_alpha, dA, lda, dB, ldb, &h_beta, dC, ldc));

        CHECK_HIP_ERROR(hC.transfer_from(dC));

        // ROCBLAS rocblas_pointer_mode_device
        CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_device));
        CHECK_HIP_ERROR(dC.transfer_from(hC_gold));
        CHECK_HIP_ERROR(hipMemcpy(d_alpha, &h_alpha, sizeof(T), hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(d_beta, &h_beta, sizeof(T), hipMemcpyHostToDevice));
        CHECK_ROCBLAS_ERROR(rocblas_gemm<T>(
            handle, transA, transB, M, N, K, d_alpha, dA, lda, dB, ldb, d_beta, dC, ldc));

        if(multi_device > 1 && deviceId!=0)
        {
            memBarrier2.wait(deviceId);
        }

        if(multi_device==1 || (multi_device > 1 && deviceId==0))
        {
            cpu_time_used = get_time_us();
            
            blis_gemm<T>(transA,
                        transB,
                        M,
                        N,
                        K,
                        h_alpha,
                        hA.data(),
                        lda,
                        hB.data(),
                        ldb,
                        h_beta,
                        hC_gold.data(),
                        ldc);

            cpu_time_used = get_time_us() - cpu_time_used;
            cblas_gflops  = gemm_gflop_count<T>(M, N, K) / cpu_time_used * 1e6;

            if(multi_device > 1)
            {
                memBarrier2.wait(deviceId);
            }
        }

        //releasing already used host memory
        hA=host_matrix<T>();
        hB=host_matrix<T>();

        for(int i = 0; i<2; i++)
        {
            if(arg.unit_check)
            {
                if(std::is_same_v<T, rocblas_half> && K > 10000)
                {
                    // For large K, rocblas_half tends to diverge proportional to K
                    // Tolerance is slightly greater than 1 / 1024.0
                    const double tol = K * sum_error_tolerance<T>;
                    near_check_general<T>(M, N, ldc, (T*)hC_gold, (T*)hC, tol);
                }
                else
                {
                    unit_check_general<T>(M, N, ldc, (T*)hC_gold, (T*)hC);
                }
            }

            if(arg.norm_check)
            {
                auto err1     = fabs(norm_check_general<T>('F', M, N, ldc, (T*)hC_gold, (T*)hC));
                rocblas_error = err1 > rocblas_error ? err1 : rocblas_error;
            }

            if(i==0)
                CHECK_HIP_ERROR(hC.transfer_from(dC));
        }

        hC=host_matrix<T>();
    }
#endif

    int number_cold_calls = 2;
    int number_hot_calls  = arg.iters;
    hipEvent_t start, stop, flush;
    CHECK_HIP_ERROR(hipEventCreateWithFlags(&flush, hipEventReleaseToSystem));
    CHECK_HIP_ERROR(hipEventCreate(&start));
    CHECK_HIP_ERROR(hipEventCreate(&stop));
    float kernel_time = 0.0f;
    host_time        = 0.0;
    float kernel_time_iter = 0.0f;
    double host_time_iter = 0.0f;

    CHECK_ROCBLAS_ERROR(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));

    for(int i = 0; i < number_cold_calls; i++)
    {
        rocblas_gemm<T>(
            handle, transA, transB, M, N, K, &h_alpha, dA, lda, dB, ldb, &h_beta, dC, ldc);
    }
    

    if(time_each_iter)
    {
        for(int i = 0; i < number_hot_calls; i++)
        {
            if(reinit_c && ((arg.norm_check && i == 0) || i > 0))
                CHECK_HIP_ERROR(dC.transfer_from(hC_orig));
            if(arg.flush_gpu_cache)
                CHECK_HIP_ERROR(hipEventRecord(flush, NULL));

            host_time_iter = get_time_us();
            CHECK_HIP_ERROR(hipEventRecord(start, NULL));

            rocblas_gemm<T>(
            handle, transA, transB, M, N, K, &h_alpha, dA, lda, dB, ldb, &h_beta, dC, ldc);

            CHECK_HIP_ERROR(hipEventRecord(stop, NULL));
            CHECK_HIP_ERROR(hipEventSynchronize(stop));
            host_time += get_time_us() - host_time_iter;
            CHECK_HIP_ERROR(hipEventElapsedTime(&kernel_time_iter, start, stop));
            kernel_time+=kernel_time_iter;
        }
    }
    else
    {
        std::pair<double,double> times;
        if(multi_device>1)
        {
            usleep(0.5 * 1000000);
            perfBarrier.wait(deviceId);
        }
        times.first = get_time_us(); // in microseconds
        CHECK_HIP_ERROR(hipEventRecord(start, NULL));
        for(int i = 0; i < number_hot_calls; i++)
        {
            rocblas_gemm<T>(
            handle, transA, transB, M, N, K, &h_alpha, dA, lda, dB, ldb, &h_beta, dC, ldc);
        }

        CHECK_HIP_ERROR(hipEventRecord(stop, NULL));
        CHECK_HIP_ERROR(hipEventSynchronize(stop));
        times.second = get_time_us();
        if(multi_device>1)
            promise.set_value(times);
        CHECK_HIP_ERROR(hipEventElapsedTime(&kernel_time, start, stop));
        host_time = times.second-times.first;
    }

    if(storeOutputData)
    {
        CHECK_HIP_ERROR(hC.transfer_from(dC));
        storeOutputToBin<T>(hC, o_file);
    }

    rocblas_gflops = gemm_gflop_count<T>(M, N, K) * number_hot_calls / kernel_time * 1e3;

    std::stringstream msg;

    if(multi_device>1)
    {
        double host_gflops = gemm_gflop_count<T>(M, N, K) * number_hot_calls / (host_time) * 1e6;
        msg << "Device " << deviceId << std::endl
        << "transA,transB,M,N,K,alpha,lda,ldb,beta,ldc,rocblas-Gflops,rocblas-Gflops(using host_time),host_time(us),kernel_time(us)" << std::endl
        << arg.transA << "," << arg.transB << "," << M << "," << N << "," << K << ","
        << arg.get_alpha<T>() << "," << lda << "," << ldb << "," << arg.get_beta<T>() << "," << ldc
        << "," << rocblas_gflops << "," << host_gflops << "," << host_time / number_hot_calls << "," << kernel_time/number_hot_calls*1000 << std::endl;
    }
    else
        msg << "transA,transB,M,N,K,alpha,lda,ldb,beta,ldc,rocblas-Gflops,host_time(us),kernel_time(us)" << std::endl
        << arg.transA << "," << arg.transB << "," << M << "," << N << "," << K << ","
        << arg.get_alpha<T>() << "," << lda << "," << ldb << "," << arg.get_beta<T>() << "," << ldc
        << "," << rocblas_gflops << "," << host_time / number_hot_calls << "," << kernel_time/number_hot_calls*1000 << std::endl;

    if(arg.norm_check)
    {
        msg << "cblas-Gflops,us,rocblas-error" << std::endl
        << cblas_gflops << "," << cpu_time_used << "," << rocblas_error << std::endl;
    }

    rocblas_cout  << msg.str();
}

int launch_bench(Arguments& arg, std::promise<std::pair<double,double>> promise)
{
    if(function == "gemm")
    {
        if(precision == "f32_r" || precision == "s")
        {
            BenchGemm<float>(arg, std::move(promise));
        }
        else if(precision == "f64_r" || precision == "d")
        {
            BenchGemm<double>(arg, std::move(promise));
        }
        else if(precision == "f16_r")
        {
            BenchGemm<rocblas_half>(arg, std::move(promise));
        }
        else
        {
            rocblas_cout << "Precision not implemented, exiting";
            return rocblas_status_not_implemented;
        }
    }
    else if(function == "gemm_strided_batched")
    {
        if(precision == "f32_r" || precision == "s")
        {
            BenchGemmStridedBatched<float>(arg, std::move(promise));
        }
        else if(precision == "f64_r" || precision == "d")
        {
            BenchGemmStridedBatched<double>(arg, std::move(promise));
        }
        else if(precision == "f16_r")
        {
            BenchGemmStridedBatched<rocblas_half>(arg, std::move(promise));
        }
        else
        {
            rocblas_cout << "Precision not implemented, exiting";
            return rocblas_status_not_implemented;
        }
    }
    else if(function == "gemm_ex")
    {
        if((a_type == "f64_r" || a_type == "d") && (b_type == "f64_r" || b_type == "d")
           && (c_type == "f64_r" || c_type == "d") && (d_type == "f64_r" || d_type == "d")
           && (compute_type == "f64_r" || compute_type == "d"))
        {   
            BenchGemmEx<double, double, double>(arg, std::move(promise));
        }
        else if((a_type == "f32_r" || a_type == "s") && (b_type == "f32_r" || b_type == "s")
                && (c_type == "f32_r" || c_type == "s") && (d_type == "f32_r" || d_type == "s")
                && (compute_type == "f32_r" || compute_type == "s"))
        {
            BenchGemmEx<float, float, float>(arg, std::move(promise));
        }
        else if((a_type == "bf16_r") && (b_type == "bf16_r")
                && (c_type == "bf16_r") && (d_type == "bf16_r")
                && (compute_type == "f32_r" || compute_type == "s"))
        {
            BenchGemmEx<rocblas_bfloat16, rocblas_bfloat16, float>(arg, std::move(promise));
        }
        else if(a_type == "f16_r"  && b_type == "f16_r"
                && c_type == "f16_r" && d_type == "f16_r"
                && compute_type == "f16_r")
        {
            BenchGemmEx<rocblas_half, rocblas_half, rocblas_half>(arg, std::move(promise));
        }
        // gemm_ex: fp16->fp32
        else if(a_type == "f16_r"  && b_type == "f16_r"
                && c_type == "f32_r" && d_type == "f32_r"
                && (compute_type == "f32_r" || compute_type == "s"))
        {
            BenchGemmEx<rocblas_half, float, float>(arg, std::move(promise));
        }
        else if(a_type == "f16_r"  && b_type == "f16_r"
                && c_type == "f16_r" && d_type == "f16_r"
                && (compute_type == "f32_r" || compute_type == "s"))
        {
            BenchGemmEx<rocblas_half, rocblas_half, float>(arg, std::move(promise));
        }
        else if(a_type == "i8_r"  && b_type == "i8_r"
                && c_type == "i32_r" && d_type == "i32_r"
                && compute_type == "i32_r")
        {
            BenchGemmEx<int8_t, int32_t, int32_t>(arg, std::move(promise));
        }
        else
        {
            rocblas_cout << "Precision not implemented, exiting";
            return rocblas_status_not_implemented;
        }
    }
    else
    {
        rocblas_cout << "Function not implemented, exiting";
        return rocblas_status_not_implemented;
    }

    return 0;
}

int main(int argc, char* argv[])
{

    Arguments arg;
    readArgs(argc, argv, arg);

    if(arg.norm_check || arg.unit_check)
    {
#ifdef VALIDATE
        setup_blis();
#else
        rocblas_cout << "run ./install -v 1 to enable validation" << std::endl;
        exit(1);
#endif
    }

    auto promise = std::make_unique<std::promise<std::pair<double,double>>[]>(multi_device);

    if(multi_device>1)
    {
        std::vector<std::thread> threads;
        auto future  = std::make_unique<std::future<std::pair<double,double>>[]>(multi_device);

        for(size_t i = 0; i < multi_device; ++i)
            future[i] = promise[i].get_future();

        for(int i = 0 ; i<multi_device; ++i)
            threads.push_back(std::thread([&, i] { set_device(i); launch_bench(arg, std::move(promise[i])); }));

        perfBarrier.wait_to_trigger();

        std::vector<std::pair<double,double>> times(multi_device);

        //wait for promises
        for(size_t i = 0; i < multi_device; ++i)
            times[i] = future[i].get(); 

        double start=times[0].first;
        double end=times[0].second;

        for(int i =0; i<multi_device; i++)
        {
            if(times[i].first < start)
                start = times[i].first;
            if(times[i].second > end)
                end = times[i].second;
        }

        for(int i =0; i<multi_device; i++)
            threads[i].join();

        //print overall run data
        double overall_time = (end-start)/arg.iters;

        double overall_gflops;
        if(arg.d_type == rocblas_datatype_f16_r)
            overall_gflops = gemm_gflop_count<rocblas_half>(arg.M, arg.N, arg.K);
        else if(arg.d_type == rocblas_datatype_bf16_r)
            overall_gflops = gemm_gflop_count<rocblas_bfloat16>(arg.M, arg.N, arg.K);
        else if(arg.d_type == rocblas_datatype_f32_r)
            overall_gflops = gemm_gflop_count<float>(arg.M, arg.N, arg.K);
        else if(arg.d_type == rocblas_datatype_f64_r)
            overall_gflops = gemm_gflop_count<double>(arg.M, arg.N, arg.K);
        else
        {
            rocblas_cout << "Precision not implemented, exiting";
            return rocblas_status_not_implemented;
        }
        overall_gflops *= arg.batch_count;
        overall_gflops /= overall_time / 1e6 / multi_device; 

        rocblas_cout<<"Overall performance using host timing"<<std::endl
        << "transA,transB,M,N,K,alpha,lda,ldb,beta,ldc,rocblas-Gflops,host_time(us)"<< std::endl
        << arg.transA << "," << arg.transB << ","
        << arg.M << "," << arg.N << "," << arg.K << "," << arg.alpha << "," << arg.lda << "," << arg.ldb
        << "," << arg.beta << "," << arg.ldc  << "," << overall_gflops << ","
        << overall_time  << std::endl;
    }
    else
    {
        return launch_bench(arg, std::move(promise[0]));
    }


    return 0;
}
