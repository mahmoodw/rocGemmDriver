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
#ifndef _UTILITY_
#define _UTILITY_

#include "rocblas.h"
#include <cmath>
#include <fstream>
#include <immintrin.h>
#include <iomanip>
#include <iostream>
#include <random>
#include <sys/time.h>
#include <vector>
#include <mutex>
#include <condition_variable>
#include <future>
#include <queue>
#include <fcntl.h>
#include <type_traits>
#include <sys/stat.h>

#include <boost/program_options.hpp>
namespace po = boost::program_options;

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
    case rocblas_datatype_invalid: return "invalid";
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

#define rocblas_cout (internal_ostream::cout())
#define rocblas_cerr (internal_ostream::cerr())

/***************************************************************************
 * The internal_ostream class performs atomic IO on log files, and provides *
 * consistent formatting                                                   *
 ***************************************************************************/
class internal_ostream
{
    /**************************************************************************
     * The worker class sets up a worker thread for writing to log files. Two *
     * files are considered the same if they have the same device ID / inode. *
     **************************************************************************/
    class worker
    {
        // task_t represents a payload of data and a promise to finish
        class task_t
        {
            std::string        str;
            std::promise<void> promise;

        public:
            // The task takes ownership of the string payload
            task_t(std::string&& str, std::promise<void>&& promise)
                : str(std::move(str))
                , promise(std::move(promise))
            {
            }

            auto get_future()
            {
                return promise.get_future();
            }

            // Notify the future to wake up
            void set_value()
            {
                promise.set_value();
            }

            // Size of the string payload
            size_t size() const
            {
                return str.size();
            }

            // Data of the string payload
            const char* data() const
            {
                return str.data();
            }
        };

        // FILE is used for safety in the presence of signals
        FILE* file = nullptr;

        // This worker's thread
        std::thread thread;

        // Condition variable for worker notification
        std::condition_variable cond;

        // Mutex for this thread's queue
        std::mutex mutex;

        // Queue of tasks
        std::queue<task_t> queue;

        // Worker thread which waits for and handles tasks sequentially
        void thread_function()
        {
            // Clear any errors in the FILE
            clearerr(file);

            // Lock the mutex in preparation for cond.wait
            std::unique_lock<std::mutex> lock(mutex);

            while(true)
            {
                // Wait for any data, ignoring spurious wakeups
                cond.wait(lock, [&] { return !queue.empty(); });

                // With the mutex locked, get and pop data from the front of queue
                task_t task = std::move(queue.front());
                queue.pop();

                // Temporarily unlock queue mutex, unblocking other threads
                lock.unlock();

                // An empty message indicates the closing of the stream
                if(!task.size())
                {
                    // Tell future to wake up
                    task.set_value();
                    break;
                }

                // Write the data
                fwrite(task.data(), 1, task.size(), file);

                // Detect any error and flush the C FILE stream
                if(ferror(file) || fflush(file))
                {
                    perror("Error writing log file");

                    // Tell future to wake up
                    task.set_value();
                    break;
                }

                // Promise that the data has been written
                task.set_value();

                // Re-lock the mutex in preparation for cond.wait
                lock.lock();
            }
        }

    public:
        // Worker constructor creates a worker thread for a raw filehandle
        explicit worker(int fd)
        {
            // The worker duplicates the file descriptor (RAII)
            fd = fcntl(fd, F_DUPFD_CLOEXEC, 0);

            // If the dup fails or fdopen fails, print error and abort
            if(fd == -1 || !(file = fdopen(fd, "a")))
            {
                perror("fdopen() error");
                rocblas_abort();
            }

            // Create a worker thread, capturing *this
            thread = std::thread([=] { thread_function(); });

            // Detatch from the worker thread
            thread.detach();
        }

        // Send a string to be written
        void send(std::string str)
        {
            // Create a promise to wait for the operation to complete
            std::promise<void> promise;

            // The future indicating when the operation has completed
            auto future = promise.get_future();

            // task_t consists of string and promise
            // std::move transfers ownership of str and promise to task
            task_t worker_task(std::move(str), std::move(promise));

            // Submit the task to the worker assigned to this device/inode
            // Hold mutex for as short as possible, to reduce contention
            // TODO: Consider whether notification should be done with lock held or released
            {
                std::lock_guard<std::mutex> lock(mutex);
                queue.push(std::move(worker_task));
                cond.notify_one();
            }

            // Wait for the task to be completed, to ensure flushed IO
            future.get();
        }

        // Destroy a worker when all std::shared_ptr references to it are gone
        ~worker()
        {
            // Tell worker thread to exit, by sending it an empty string
            send({});

            // Close the FILE
            if(file)
                fclose(file);
        }
    };

    // Two filehandles point to the same file if they share the same (std_dev, std_ino).
    
    // Initial slice of struct stat which contains device ID and inode
    struct file_id_t
    {
        dev_t st_dev; // ID of device containing file
        ino_t st_ino; // Inode number
    };

    // Compares device IDs and inodes for map containers
    struct file_id_less
    {
        bool operator()(const file_id_t& lhs, const file_id_t& rhs) const
        {
            return lhs.st_ino < rhs.st_ino || (lhs.st_ino == rhs.st_ino && lhs.st_dev < rhs.st_dev);
        }
    };

    // Map from file_id to a worker shared_ptr
    // Implemented as singleton to avoid the static initialization order fiasco
    static auto& map()
    {
        static std::map<file_id_t, std::shared_ptr<worker>, file_id_less> map;
        return map;
    }

    // Mutex for accessing the map
    // Implemented as singleton to avoid the static initialization order fiasco
    static auto& map_mutex()
    {
        static std::recursive_mutex map_mutex;
        return map_mutex;
    }

    // Map from file_id to a worker shared_ptr
    // Implemented as singleton to avoid the static initialization order fiasco
    static auto& worker_map()
    {
        static std::map<file_id_t, std::shared_ptr<worker>, file_id_less> file_id_to_worker_map;
        return file_id_to_worker_map;
    }

    // Mutex for accessing the map
    // Implemented as singleton to avoid the static initialization order fiasco
    static auto& worker_map_mutex()
    {
        static std::recursive_mutex map_mutex;
        return map_mutex;
    }

    // Output buffer for formatted IO
    std::ostringstream os;

    // Worker thread for accepting tasks
    std::shared_ptr<worker> worker_ptr;

    // Flag indicating whether YAML mode is turned on
    bool yaml = false;

    // Get worker for file descriptor
    static std::shared_ptr<worker> get_worker(int fd)
    {
        // For a file descriptor indicating an error, return a nullptr
        if(fd == -1)
            return nullptr;

        // C++ allows type punning of common initial sequences
        union
        {
            struct stat statbuf;
            file_id_t   file_id;
        };

        // Verify common initial sequence
        static_assert(std::is_standard_layout<file_id_t>{} && std::is_standard_layout<struct stat>{}
                        && offsetof(file_id_t, st_dev) == 0 && offsetof(struct stat, st_dev) == 0
                        && offsetof(file_id_t, st_ino) == offsetof(struct stat, st_ino)
                        && std::is_same_v<decltype(file_id_t::st_dev), decltype(stat::st_dev)>
                        && std::is_same_v<decltype(file_id_t::st_ino), decltype(stat::st_ino)>,
                    "struct stat and file_id_t are not layout-compatible");

        // Get the device ID and inode, to detect common files
        if(fstat(fd, &statbuf))
        {
            perror("Error executing fstat()");
            return nullptr;
        }

        // Lock the map from file_id -> std::shared_ptr<internal_ostream::worker>
        std::lock_guard<std::recursive_mutex> lock(map_mutex());

        // Insert a nullptr map element if file_id doesn't exist in map already
        // worker_ptr is a reference to the std::shared_ptr<internal_ostream::worker>
        auto& worker_ptr = map().emplace(file_id, nullptr).first->second;

        // If a new entry was inserted, or an old entry is empty, create new worker
        if(!worker_ptr)
            worker_ptr = std::make_shared<worker>(fd);

        // Return the existing or new worker matching the file
        return worker_ptr;
    }

    // Private explicit copy constructor duplicates the worker and starts a new buffer
    explicit internal_ostream(const internal_ostream& other)
        : worker_ptr(other.worker_ptr)
    {
    }

public:
    // Default constructor is a std::ostringstream with no worker
    internal_ostream() = default;

    // Move constructor
    internal_ostream(internal_ostream&&) = default;

    // Move assignment
    internal_ostream& operator=(internal_ostream&&) & = default;

    // Copy assignment is deleted
    internal_ostream& operator=(const internal_ostream&) = delete;

    // Construct from a file descriptor, which is duped
    explicit internal_ostream(int fd)
        : worker_ptr(get_worker(fd))
    {
        if(!worker_ptr)
        {
            dprintf(STDERR_FILENO, "Error: Bad file descriptor %d\n", fd);
            rocblas_abort();
        }
    }

    // Construct from a C filename
    explicit internal_ostream(const char* filename)
    {
        int fd     = open(filename, O_WRONLY | O_CREAT | O_TRUNC | O_APPEND | O_CLOEXEC, 0644);
        worker_ptr = get_worker(fd);
        if(!worker_ptr)
        {
            dprintf(STDERR_FILENO, "Cannot open %s: %m\n", filename);
            rocblas_abort();
        }
        close(fd);
    }

    // Construct from a std::string filename
    explicit internal_ostream(const std::string& filename)
        : internal_ostream(filename.c_str())
    {
    }

    // Create a duplicate of this
    internal_ostream dup() const
    {
        if(!worker_ptr)
            throw std::runtime_error(
                "Attempting to duplicate a internal_ostream without an associated file");
        return internal_ostream(*this);
    }

    // For testing to allow file closing and deletion
    static void clear_workers();

    // Convert stream output to string
    std::string str() const
    {
        return os.str();
    }

    // Clear the buffer
    void clear()
    {
        os.clear();
        os.str({});
    }

    // Flush the output
    void flush()
    {
        // Flush only if this stream contains a worker (i.e., is not a string)
        if(worker_ptr)
        {
            // The contents of the string buffer
            auto str = os.str();

            // Empty string buffers kill the worker thread, so they are not flushed here
            if(str.size())
                worker_ptr->send(std::move(str));

            // Clear the string buffer
            clear();
        }
    }

    // Destroy the internal_ostream
    virtual ~internal_ostream()
    {
        flush(); // Flush any pending IO
    }

    // Implemented as singleton to avoid the static initialization order fiasco
    static internal_ostream& cout()
    {
        thread_local internal_ostream t_cout{STDOUT_FILENO};
        return t_cout;
    }

    // Implemented as singleton to avoid the static initialization order fiasco
    static internal_ostream& cerr()
    {
        thread_local internal_ostream t_cerr{STDERR_FILENO};
        return t_cerr;
    }

    // Abort function which safely flushes all IO
    friend void rocblas_abort_once();

    /*************************************************************************
     * Non-member friend functions for formatted output                      *
     *************************************************************************/

    // Default output for non-enumeration types
    template <typename T, std::enable_if_t<!std::is_enum<std::decay_t<T>>{}, int> = 0>
    friend internal_ostream& operator<<(internal_ostream& os, T&& x)
    {
        os.os << std::forward<T>(x);
        return os;
    }

    // Default output for enumeration types
    template <typename T, std::enable_if_t<std::is_enum<std::decay_t<T>>{}, int> = 0>
    friend internal_ostream& operator<<(internal_ostream& os, T&& x)
    {
        os.os << std::underlying_type_t<std::decay_t<T>>(x);
        return os;
    }

    // Pairs for YAML output
    template <typename T1, typename T2>
    friend internal_ostream& operator<<(internal_ostream& os, std::pair<T1, T2> p)
    {
        os << p.first << ": ";
        os.yaml = true;
        os << p.second;
        os.yaml = false;
        return os;
    }

    // Complex output
    template <typename T>
    friend internal_ostream& operator<<(internal_ostream&     os,
                                                const rocblas_complex_num<T>& x)
    {
        if(os.yaml)
            os.os << "'(" << std::real(x) << "," << std::imag(x) << ")'";
        else
            os.os << x;
        return os;
    }

    // Floating-point output
    friend internal_ostream& operator<<(internal_ostream& os, double x)
    {
        if(!os.yaml)
            os.os << x;
        else
        {
            // For YAML, we must output the floating-point value exactly
            if(std::isnan(x))
                os.os << ".nan";
            else if(std::isinf(x))
                os.os << (x < 0 ? "-.inf" : ".inf");
            else
            {
                char s[32];
                snprintf(s, sizeof(s) - 2, "%.17g", x);

                // If no decimal point or exponent, append .0 to indicate floating point
                for(char* end = s; *end != '.' && *end != 'e' && *end != 'E'; ++end)
                {
                    if(!*end)
                    {
                        end[0] = '.';
                        end[1] = '0';
                        end[2] = '\0';
                        break;
                    }
                }
                os.os << s;
            }
        }
        return os;
    }

    friend internal_ostream& operator<<(internal_ostream& os, rocblas_half half)
    {
        return os << float(half);
    }

    friend internal_ostream& operator<<(internal_ostream& os, rocblas_bfloat16 bf16)
    {
        return os << float(bf16);
    }

    // Integer output
    friend internal_ostream& operator<<(internal_ostream& os, int32_t x)
    {
        os.os << x;
        return os;
    }
    friend internal_ostream& operator<<(internal_ostream& os, uint32_t x)
    {
        os.os << x;
        return os;
    }
    friend internal_ostream& operator<<(internal_ostream& os, int64_t x)
    {
        os.os << x;
        return os;
    }
    friend internal_ostream& operator<<(internal_ostream& os, uint64_t x)
    {
        os.os << x;
        return os;
    }

    // bool output
    friend internal_ostream& operator<<(internal_ostream& os, bool b)
    {
        if(os.yaml)
            os.os << (b ? "true" : "false");
        else
            os.os << (b ? 1 : 0);
        return os;
    }

    // Character output
    friend internal_ostream& operator<<(internal_ostream& os, char c)
    {
        if(os.yaml)
        {
            char s[]{c, 0};
            os.os << std::quoted(s, '\'');
        }
        else
            os.os << c;
        return os;
    }

    // String output
    friend internal_ostream& operator<<(internal_ostream& os, const char* s)
    {
        if(os.yaml)
            os.os << std::quoted(s);
        else
            os.os << s;
        return os;
    }

    friend internal_ostream& operator<<(internal_ostream& os, const std::string& s)
    {
        return os << s.c_str();
    }

    // rocblas_datatype output
    friend internal_ostream& operator<<(internal_ostream& os, rocblas_datatype d)
    {
        os.os << rocblas_datatype_string(d);
        return os;
    }

    // rocblas_operation output
    friend internal_ostream& operator<<(internal_ostream& os,
                                                rocblas_operation         trans)

    {
        return os << rocblas_transpose_letter(trans);
    }

    // rocblas_fill output
    friend internal_ostream& operator<<(internal_ostream& os, rocblas_fill fill)

    {
        return os << rocblas_fill_letter(fill);
    }

    // rocblas_diagonal output
    friend internal_ostream& operator<<(internal_ostream& os, rocblas_diagonal diag)

    {
        return os << rocblas_diag_letter(diag);
    }

    // rocblas_side output
    friend internal_ostream& operator<<(internal_ostream& os, rocblas_side side)

    {
        return os << rocblas_side_letter(side);
    }

    // rocblas_status output
    friend internal_ostream& operator<<(internal_ostream& os, rocblas_status status)
    {
        os.os << rocblas_status_to_string(status);
        return os;
    }

    // atomics mode output
    friend internal_ostream& operator<<(internal_ostream& os,
                                                rocblas_atomics_mode      mode)
    {
        os.os << rocblas_atomics_mode_to_string(mode);
        return os;
    }

    // gemm flags output
    friend internal_ostream& operator<<(internal_ostream& os,
                                                rocblas_gemm_flags        flags)
    {
        os.os << rocblas_gemm_flags_to_string(flags);
        return os;
    }

    // Transfer internal_ostream to std::ostream
    friend std::ostream& operator<<(std::ostream& os, const internal_ostream& str)
    {
        return os << str.str();
    }

    // Transfer internal_ostream to internal_ostream
    friend internal_ostream& operator<<(internal_ostream&       os,
                                                const internal_ostream& str)
    {
        return os << str.str();
    }

    // IO Manipulators

    friend internal_ostream& operator<<(internal_ostream& os,
                                                std::ostream& (*pf)(std::ostream&))
    {
        // Turn YAML formatting on or off
        if(pf == internal_ostream::yaml_on)
            os.yaml = true;
        else if(pf == internal_ostream::yaml_off)
            os.yaml = false;
        else
        {
            // Output the manipulator to the buffer
            os.os << pf;

            // If the manipulator is std::endl or std::flush, flush the output
            if(pf == static_cast<std::ostream& (*)(std::ostream&)>(std::endl)
               || pf == static_cast<std::ostream& (*)(std::ostream&)>(std::flush))
            {
                os.flush();
            }
        }
        return os;
    }

    // YAML Manipulators (only used for their addresses now)
    static std::ostream& yaml_on(std::ostream& os);
    static std::ostream& yaml_off(std::ostream& os);
};

#define CHECK_HIP_ERROR(ERROR)                    \
    do                                            \
    {                                             \
        auto error = ERROR;                       \
        if(error != hipSuccess)                   \
        {                                         \
            fprintf(stderr,                       \
                    "error: '%s'(%d) at %s:%d\n", \
                    hipGetErrorString(error),     \
                    error,                        \
                    __FILE__,                     \
                    __LINE__);                    \
            exit(EXIT_FAILURE);                   \
        }                                         \
    } while(0)

#define EXPECT_ROCBLAS_STATUS rocblas_expect_status

#define CHECK_ROCBLAS_ERROR2(STATUS) EXPECT_ROCBLAS_STATUS(STATUS, rocblas_status_success)
#define CHECK_ROCBLAS_ERROR(STATUS) CHECK_ROCBLAS_ERROR2(STATUS)

/*! \brief  CPU Timer(in microsecond): synchronize with the default device and return wall time */
double get_time_us(void)
{
    CHECK_HIP_ERROR(hipDeviceSynchronize());
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (tv.tv_sec * 1000 * 1000) + tv.tv_usec;
};

/* ============================================================================================ */
/*  device query and print out their ID and name; return number of compute-capable devices. */
rocblas_int query_device_property()
{
    int            device_count;
    rocblas_status status = (rocblas_status)hipGetDeviceCount(&device_count);
    if(status != rocblas_status_success)
    {
        rocblas_cerr << "Query device error: cannot get device count" << std::endl;
        return -1;
    }
    else
    {
        rocblas_cout << "Query device success: there are " << device_count << " devices"
                     << std::endl;
    }

    for(rocblas_int i = 0;; i++)
    {
        rocblas_cout
            << "-------------------------------------------------------------------------------"
            << std::endl;

        if(i >= device_count)
            break;

        hipDeviceProp_t props;
        rocblas_status  status = (rocblas_status)hipGetDeviceProperties(&props, i);
        if(status != rocblas_status_success)
        {
            rocblas_cerr << "Query device error: cannot get device ID " << i << "'s property"
                         << std::endl;
        }
        else
        {
            char buf[320];
            snprintf(
                buf,
                sizeof(buf),
                "Device ID %d : %s %s\n"
                "with %3.1f GB memory, max. SCLK %d MHz, max. MCLK %d MHz, compute capability "
                "%d.%d\n"
                "maxGridDimX %d, sharedMemPerBlock %3.1f KB, maxThreadsPerBlock %d, warpSize %d\n",
                i,
                props.name,
                props.gcnArchName,
                props.totalGlobalMem / 1e9,
                (int)(props.clockRate / 1000),
                (int)(props.memoryClockRate / 1000),
                props.major,
                props.minor,
                props.maxGridSize[0],
                props.sharedMemPerBlock / 1e3,
                props.maxThreadsPerBlock,
                props.warpSize);
            rocblas_cout << buf;
        }
    }

    return device_count;
}

/*  set current device to device_id */
void set_device(rocblas_int device_id)
{
    rocblas_status status = (rocblas_status)hipSetDevice(device_id);
    if(status != rocblas_status_success)
    {
        printf("Set device error: cannot set device ID %d, there may not be such device ID\n",
               (int)device_id);
    }
}

inline const char* rocblas_status_to_string(rocblas_status status)
{
    switch(status)
    {
    case rocblas_status_success:
        return "rocblas_status_success";
    case rocblas_status_invalid_handle:
        return "rocblas_status_invalid_handle";
    case rocblas_status_not_implemented:
        return "rocblas_status_not_implemented";
    case rocblas_status_invalid_pointer:
        return "rocblas_status_invalid_pointer";
    case rocblas_status_invalid_size:
        return "rocblas_status_invalid_size";
    case rocblas_status_memory_error:
        return "rocblas_status_memory_error";
    case rocblas_status_internal_error:
        return "rocblas_status_internal_error";
    default:
        return "<undefined rocblas_status value>";
    }
}

inline void rocblas_expect_status(rocblas_status status, rocblas_status expect)
{
    if(status != expect)
    {
        std::cerr << "rocBLAS status error: Expected " << rocblas_status_to_string(expect)
                  << ", received " << rocblas_status_to_string(status) << std::endl;
        if(expect == rocblas_status_success)
            exit(EXIT_FAILURE);
    }
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

/* ============================================================================================ */
// Random number generator
using rocblas_rng_t = std::mt19937;
extern rocblas_rng_t rocblas_rng, rocblas_seed;

rocblas_rng_t rocblas_rng, rocblas_seed;

// Reset the seed (mainly to ensure repeatability of failures in a given suite)
inline void rocblas_seedrand()
{
    rocblas_rng = rocblas_seed;
}

/* ============================================================================================ */
/*! \brief  Random number generator which generates NaN values */
class rocblas_nan_rng
{
    // Generate random NaN values
    template <typename T, typename UINT_T, int SIG, int EXP>
    static T random_nan_data()
    {
        static_assert(sizeof(UINT_T) == sizeof(T), "Type sizes do not match");
        union u_t
        {
            u_t() {}
            UINT_T u;
            T      fp;
        } x;
        do
            x.u = std::uniform_int_distribution<UINT_T>{}(rocblas_rng);
        while(!(x.u & (((UINT_T)1 << SIG) - 1))); // Reject Inf (mantissa == 0)
        x.u |= (((UINT_T)1 << EXP) - 1) << SIG; // Exponent = all 1's
        return x.fp; // NaN with random bits
    }

public:
    // Random integer
    template <typename T, typename std::enable_if<std::is_integral<T>{}, int>::type = 0>
    explicit operator T()
    {
        return std::uniform_int_distribution<T>{}(rocblas_rng);
    }

    // Random NaN double
    explicit operator double()
    {
        return random_nan_data<double, uint64_t, 52, 11>();
    }

    // Random NaN float
    explicit operator float()
    {
        return random_nan_data<float, uint32_t, 23, 8>();
    }

    // Random NaN half (non-template rocblas_half takes precedence over integer template above)
    explicit operator rocblas_half()
    {
        return random_nan_data<rocblas_half, uint16_t, 10, 5>();
    }

    // Random NaN bfloat16
    explicit operator rocblas_bfloat16()
    {
        return random_nan_data<rocblas_bfloat16, uint16_t, 7, 8>();
    }

    explicit operator rocblas_float_complex()
    {
        return {float(*this), float(*this)};
    }

    explicit operator rocblas_double_complex()
    {
        return {double(*this), double(*this)};
    }
};

/* ============================================================================================ */
// Helper function to truncate float to bfloat16

inline __host__ rocblas_bfloat16 float_to_bfloat16_truncate(float val)
{
    union
    {
        float    fp32;
        uint32_t int32;
    } u = {val};
    rocblas_bfloat16 ret;
    ret.data = uint16_t(u.int32 >> 16);
    if((u.int32 & 0x7fff0000) == 0x7f800000 && u.int32 & 0xffff)
        ret.data |= 1; // Preserve signaling NaN
    return ret;
}

/* ============================================================================================ */
/*! \brief  returns true if value is NaN */

template <typename T, typename std::enable_if<std::is_integral<T>{}, int>::type = 0>
inline bool rocblas_isnan(T)
{
    return false;
}

template <typename T,
          typename std::enable_if<!std::is_integral<T>{} && !rocblas_is_complex<T>, int>::type = 0>
inline bool rocblas_isnan(T arg)
{
    return std::isnan(arg);
}

template <typename T, typename std::enable_if<rocblas_is_complex<T>, int>::type = 0>
inline bool rocblas_isnan(const T& arg)
{
    return rocblas_isnan(std::real(arg)) || rocblas_isnan(std::imag(arg));
}

inline bool rocblas_isnan(rocblas_half arg)
{
    union
    {
        rocblas_half fp;
        uint16_t     data;
    } x = {arg};
    return (~x.data & 0x7c00) == 0 && (x.data & 0x3ff) != 0;
}

/* ============================================================================================ */
/*! \brief negate a value */

template <class T>
inline T negate(T x)
{
    return -x;
}

template <>
inline rocblas_half negate(rocblas_half arg)
{
    union
    {
        rocblas_half fp;
        uint16_t     data;
    } x = {arg};

    x.data ^= 0x8000;
    return x.fp;
}

template <>
inline rocblas_bfloat16 negate(rocblas_bfloat16 x)
{
    x.data ^= 0x8000;
    return x;
}

/* ============================================================================================ */
/* generate random number :*/

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

/*! \brief  generate a random number in HPL-like [-0.5,0.5] doubles  */
template <typename T>
inline T random_hpl_generator()
{
    return std::uniform_real_distribution<double>(-0.5, 0.5)(rocblas_rng);
}

template <>
inline rocblas_bfloat16 random_hpl_generator()
{
    return rocblas_bfloat16(std::uniform_real_distribution<double>(-0.5, 0.5)(rocblas_rng));
}

#define MEM_MAX_GUARD_PAD 8192

/* ============================================================================================ */
/*! \brief  base-class to allocate/deallocate device memory */
template <typename T>
class d_vector
{
private:
    size_t m_size;
    size_t m_pad, m_guard_len;
    size_t m_bytes;

    static bool m_init_guard;

public:
    inline size_t nmemb() const noexcept
    {
        return m_size;
    }

public:
    bool use_HMM = false;

public:
    static T m_guard[MEM_MAX_GUARD_PAD];

#ifdef GOOGLE_TEST
    d_vector(size_t s, bool HMM = false)
        : m_size(s)
        , m_pad(std::min(g_DVEC_PAD, size_t(MEM_MAX_GUARD_PAD)))
        , m_guard_len(m_pad * sizeof(T))
        , m_bytes((s + m_pad * 2) * sizeof(T))
        , use_HMM(HMM)
    {
        // Initialize m_guard with random data
        if(!m_init_guard)
        {
            rocblas_init_nan(m_guard, MEM_MAX_GUARD_PAD);
            m_init_guard = true;
        }
    }
#else
    d_vector(size_t s, bool HMM = false)
        : m_size(s)
        , m_pad(0) // save current pad length
        , m_guard_len(0 * sizeof(T))
        , m_bytes(s ? s * sizeof(T) : sizeof(T))
        , use_HMM(HMM)
    {
    }
#endif

    T* device_vector_setup()
    {
        T* d = nullptr;
        if(use_HMM ? hipMallocManaged(&d, m_bytes) : (hipMalloc)(&d, m_bytes) != hipSuccess)
        {
            rocblas_cerr << "Warning: hip can't allocate " << m_bytes << " bytes ("
                         << (m_bytes >> 30) << " GB)" << std::endl;

            d = nullptr;
        }
#ifdef GOOGLE_TEST
        else
        {
            if(m_guard_len > 0)
            {
                // Copy m_guard to device memory before allocated memory
                if(hipMemcpy(d, m_guard, m_guard_len, hipMemcpyDefault) != hipSuccess)
                    rocblas_cerr << "Error: hipMemcpy pre-guard copy failure." << std::endl;

                // Point to allocated block
                d += m_pad;

                // Copy m_guard to device memory after allocated memory
                if(hipMemcpy(d + m_size, m_guard, m_guard_len, hipMemcpyDefault) != hipSuccess)
                    rocblas_cerr << "Error: hipMemcpy post-guard copy failure." << std::endl;
            }
        }
#endif
        return d;
    }

    void device_vector_check(T* d)
    {
#ifdef GOOGLE_TEST
        if(m_pad > 0)
        {
            T host[m_pad];

            // Copy device memory after allocated memory to host
            if(hipMemcpy(host, d + this->m_size, m_guard_len, hipMemcpyDefault) != hipSuccess)
                rocblas_cerr << "Error: hipMemcpy post-guard copy failure." << std::endl;

            // Make sure no corruption has occurred
            EXPECT_EQ(memcmp(host, m_guard, m_guard_len), 0);

            // Point to m_guard before allocated memory
            d -= m_pad;

            // Copy device memory after allocated memory to host
            if(hipMemcpy(host, d, m_guard_len, hipMemcpyDefault) != hipSuccess)
                rocblas_cerr << "Error: hipMemcpy pre-guard copy failure." << std::endl;

            // Make sure no corruption has occurred
            EXPECT_EQ(memcmp(host, m_guard, m_guard_len), 0);
        }
#endif
    }

    void device_vector_teardown(T* d)
    {
        if(d != nullptr)
        {
            device_vector_check(d);

            if(m_pad > 0)
                d -= m_pad; // restore to start of alloc

            // Free device memory
            CHECK_HIP_ERROR((hipFree)(d));
        }
    }
};

template <typename T>
T d_vector<T>::m_guard[MEM_MAX_GUARD_PAD] = {};

template <typename T>
bool d_vector<T>::m_init_guard = false;

//
// Forward declaration of the host vector.
//
template <typename T>
class host_vector;

//!
//! @brief pseudo-vector subclass which uses device memory
//!
template <typename T>
class device_vector : public d_vector<T>
{

public:
    //!
    //! @brief Disallow copying.
    //!
    device_vector(const device_vector&) = delete;

    //!
    //! @brief Disallow assigning
    //!
    device_vector& operator=(const device_vector&) = delete;

    //!
    //! @brief Constructor.
    //! @param n The length of the vector.
    //! @param inc Element index increment. If zero treated as one.
    //! @param HMM HipManagedMemory Flag.
    //!
    explicit device_vector(size_t n, int64_t inc = 1, bool HMM = false)
        : d_vector<T>{calculate_nmemb(n, inc), HMM}
        , m_n{n}
        , m_inc{inc ? inc : 1}
        , m_data{this->device_vector_setup()}
    {
    }

    //!
    //! @brief Destructor.
    //!
    ~device_vector()
    {
        this->device_vector_teardown(m_data);
        m_data = nullptr;
    }

    //!
    //! @brief Returns the length of the vector.
    //!
    size_t n() const
    {
        return m_n;
    }

    //!
    //! @brief Returns the increment of the vector.
    //!
    int64_t inc() const
    {
        return m_inc;
    }

    //!
    //! @brief Returns the batch count (always 1).
    //!
    int64_t batch_count() const
    {
        return 1;
    }

    //!
    //! @brief Returns the stride (out of context, always 0)
    //!
    rocblas_stride stride() const
    {
        return 0;
    }

    //!
    //! @brief Decay into pointer wherever pointer is expected.
    //!
    operator T*()
    {
        return m_data;
    }

    //!
    //! @brief Decay into constant pointer wherever pointer is expected.
    //!
    operator const T*() const
    {
        return m_data;
    }

    //!
    //! @brief Transfer data from a host vector.
    //! @param that The host vector.
    //! @return the hip error.
    //!
    hipError_t transfer_from(const host_vector<T>& that)
    {
        return hipMemcpy(m_data,
                         (const T*)that,
                         this->nmemb() * sizeof(T),
                         this->use_HMM ? hipMemcpyHostToHost : hipMemcpyHostToDevice);
    }

    hipError_t memcheck() const
    {
        return !this->nmemb() || m_data ? hipSuccess : hipErrorOutOfMemory;
    }

private:
    size_t  m_n{};
    int64_t m_inc{};
    T*      m_data{};

    static size_t calculate_nmemb(size_t n, int64_t inc)
    {
        // alllocate when n is zero
        return 1 + ((n ? n : 1) - 1) * std::abs(inc ? inc : 1);
    }
};

// light weight memory tracking for threshold limit on total use
static size_t                  mem_used{0};
static std::map<void*, size_t> mem_allocated;
static std::mutex              mem_mutex;

inline void alloc_ptr_use(void* ptr, size_t size)
{
    std::lock_guard<std::mutex> lock(mem_mutex);
    if(ptr)
    {
        mem_allocated[ptr] = size;
        mem_used += size;
    }
}

inline void free_ptr_use(void* ptr)
{
    std::lock_guard<std::mutex> lock(mem_mutex);
    if(ptr && mem_allocated[ptr])
    {
        mem_used -= mem_allocated[ptr];
        mem_allocated.erase(ptr);
    }
}

//!
//! @brief Host free memory w/o swap.  Returns kB or -1 if unknown.
//!
ptrdiff_t host_bytes_available()
{
#ifdef WIN32

    MEMORYSTATUSEX status;
    status.dwLength = sizeof(status);
    GlobalMemoryStatusEx(&status);
    return (ptrdiff_t)status.ullAvailPhys;

#else

    const int BUF_MAX = 1024;
    char      buf[BUF_MAX];

    ptrdiff_t n_bytes = -1; // unknown

    FILE* fp = popen("cat /proc/meminfo", "r");
    if(fp == NULL)
    {
        return n_bytes;
    }

    static const char* mem_token     = "MemFree";
    static auto*       mem_free_type = getenv("ROCBLAS_CLIENT_ALLOC_AVAILABLE");
    if(mem_free_type)
    {
        mem_token = "MemAvail"; // MemAvailable
    }
    int mem_token_len = strlen(mem_token);

    while(fgets(buf, BUF_MAX, fp) != NULL)
    {
        // set env ROCBLAS_CLIENT_ALLOC_AVAILABLE to use MemAvailable if too many SKIPS occur
        if(!strncmp(buf, mem_token, mem_token_len))
        {
            sscanf(buf, "%*s %td", &n_bytes); // kB assumed as 3rd column and ignored
            n_bytes *= 1024;
            break;
        }
    }

    int status = pclose(fp);
    if(status == -1)
    {
        return -1;
    }
    else
    {
        return n_bytes;
    }

#endif
}

//!
//! @brief Return rough estimate of memory used via host_ helper APIs only.
//!
size_t host_bytes_allocated()
{
    std::lock_guard<std::mutex> lock(mem_mutex);
    return mem_used;
}

inline bool host_mem_safe(size_t n_bytes)
{
#if defined(ROCBLAS_BENCH)
    return true; // roll out to rocblas-bench when CI does perf testing
#else
    static auto* no_alloc_check = getenv("ROCBLAS_CLIENT_NO_ALLOC_CHECK");
    if(no_alloc_check)
    {
        return true;
    }

    constexpr size_t threshold = 100 * 1024 * 1024; // 100 MB

    static size_t client_ram_limit = 0;

    static int once = [&] {
        auto* alloc_limit = getenv("ROCBLAS_CLIENT_RAM_GB_LIMIT");
        if(alloc_limit)
        {
            size_t mem_limit;
            client_ram_limit = sscanf(alloc_limit, "%zu", &mem_limit) == 1 ? mem_limit : 0;
            client_ram_limit <<= 30; // B to GB
        }
        return 0;
    }();

    if(n_bytes > threshold)
    {
        if(client_ram_limit)
        {
            if(host_bytes_allocated() + n_bytes > client_ram_limit)
            {
                rocblas_cerr << "Warning: skipped allocating " << n_bytes << " bytes ("
                             << (n_bytes >> 30) << " GB) as total would be more than client limit ("
                             << (client_ram_limit >> 30) << " GB)" << std::endl;

                return false;
            }
        }

        ptrdiff_t avail_bytes = host_bytes_available(); // negative if unknown
        if(avail_bytes >= 0 && n_bytes > avail_bytes)
        {
            rocblas_cerr << "Warning: skipped allocating " << n_bytes << " bytes ("
                         << (n_bytes >> 30) << " GB) as more than free memory ("
                         << (avail_bytes >> 30) << " GB)" << std::endl;

            // we don't try if it looks to push load into swap
            return false;
        }
    }
    return true;
#endif
}

//!
//! @brief Allocates memory which can be freed with free.  Returns nullptr if swap required.
//!
void* host_malloc(size_t size)
{
    if(host_mem_safe(size))
    {
        void* ptr = malloc(size);

        static int value = -1;

        static auto once = false;
        if(!once)
        {
            auto* alloc_byte_str = getenv("ROCBLAS_CLIENT_ALLOC_FILL_HEX_BYTE");
            if(alloc_byte_str)
            {
                value = strtol(alloc_byte_str, nullptr, 16); // hex
            }
            once = true;
        }

        if(value != -1 && ptr)
            memset(ptr, value, size);

        alloc_ptr_use(ptr, size);

        return ptr;
    }
    else
        return nullptr;
}

//!
//! @brief Allocates memory which can be freed with free.  Throws exception if swap required.
//!
inline void* host_malloc_throw(size_t nmemb, size_t size)
{
    void* ptr = host_malloc(nmemb * size);
    if(!ptr)
    {
        throw std::bad_alloc{};
    }
    return ptr;
}

//!
//! @brief Allocates cleared memory which can be freed with free.  Returns nullptr if swap required.
//!
void* host_calloc(size_t nmemb, size_t size)
{
    if(host_mem_safe(nmemb * size))
    {
        void* ptr = calloc(nmemb, size);
        alloc_ptr_use(ptr, size);
        return ptr;
    }
    else
        return nullptr;
}

//!
//! @brief Allocates cleared memory which can be freed with free.  Throws exception if swap required.
//!
inline void* host_calloc_throw(size_t nmemb, size_t size)
{
    void* ptr = host_calloc(nmemb, size);
    if(!ptr)
    {
        throw std::bad_alloc{};
    }
    return ptr;
}

//!
//! @brief Release memory allocated with host_ prefixed allocators
//!
void host_free(void* ptr)
{
    free(ptr);
    free_ptr_use(ptr);
}


//!
//! @brief  Allocator which allocates with host_calloc
//!
template <class T>
struct host_memory_allocator
{
    using value_type = T;

    host_memory_allocator() = default;

    template <class U>
    host_memory_allocator(const host_memory_allocator<U>&)
    {
    }

    T* allocate(std::size_t n)
    {
        return (T*)host_malloc_throw(n, sizeof(T));
    }

    void deallocate(T* ptr, std::size_t n)
    {
        host_free(ptr);
    }
};

template <class T, class U>
constexpr bool operator==(const host_memory_allocator<T>&, const host_memory_allocator<U>&)
{
    return true;
}

template <class T, class U>
constexpr bool operator!=(const host_memory_allocator<T>&, const host_memory_allocator<U>&)
{
    return false;
}

//!
//! @brief  Pseudo-vector subclass which uses host memory.
//!
template <typename T>
struct host_vector : std::vector<T, host_memory_allocator<T>>
{
    // Inherit constructors
    using std::vector<T, host_memory_allocator<T>>::vector;

    //!
    //! @brief Constructor.
    //! @param  inc Element index increment. If zero treated as one
    //!
    host_vector(size_t n, int64_t inc = 1)
        : std::vector<T, host_memory_allocator<T>>(calculate_nmemb(n, inc))
        , m_n(n)
        , m_inc(inc ? inc : 1)
    {
    }

    //!
    //! @brief Copy constructor from host_vector of other types convertible to T
    //!
    template <typename U, std::enable_if_t<std::is_convertible<U, T>{}, int> = 0>
    host_vector(const host_vector<U>& x)
        : std::vector<T, host_memory_allocator<T>>(x.size())
        , m_n(x.size())
        , m_inc(1)
    {
        for(size_t i = 0; i < m_n; ++i)
            (*this)[i] = x[i];
    }

    //!
    //! @brief Decay into pointer wherever pointer is expected
    //!
    operator T*()
    {
        return this->data();
    }

    //!
    //! @brief Decay into constant pointer wherever constant pointer is expected
    //!
    operator const T*() const
    {
        return this->data();
    }

    //!
    //! @brief Transfer from a device vector.
    //! @param  that That device vector.
    //! @return the hip error.
    //!
    hipError_t transfer_from(const device_vector<T>& that)
    {
        hipError_t hip_err;

        if(that.use_HMM && hipSuccess != (hip_err = hipDeviceSynchronize()))
            return hip_err;

        return hipMemcpy(*this,
                         that,
                         sizeof(T) * this->size(),
                         that.use_HMM ? hipMemcpyHostToHost : hipMemcpyDeviceToHost);
    }

    //!
    //! @brief Returns the length of the vector.
    //!
    size_t n() const
    {
        return m_n;
    }

    //!
    //! @brief Returns the increment of the vector.
    //!
    int64_t inc() const
    {
        return m_inc;
    }

    //!
    //! @brief Returns the batch count (always 1).
    //!
    static constexpr rocblas_int batch_count()
    {
        return 1;
    }

    //!
    //! @brief Returns the stride (out of context, always 0)
    //!
    static constexpr rocblas_stride stride()
    {
        return 0;
    }

    //!
    //! @brief Check if memory exists (out of context, always hipSuccess)
    //!
    static constexpr hipError_t memcheck()
    {
        return hipSuccess;
    }

private:
    size_t  m_n   = 0;
    int64_t m_inc = 0;

    static size_t calculate_nmemb(size_t n, int64_t inc)
    {
        return 1 + ((n ? n : 1) - 1) * std::abs(inc ? inc : 1);
    }
};

/*************************************************************************************************************************
//! @brief enum to check the type of matrix
 ************************************************************************************************************************/
typedef enum rocblas_check_matrix_type_
{
    // General matrix
    rocblas_client_general_matrix,

    // Hermitian matrix
    rocblas_client_hermitian_matrix,

    // Symmetric matrix
    rocblas_client_symmetric_matrix,

    // Triangular matrix
    rocblas_client_triangular_matrix,

    // Diagonally dominant triangular matrix
    rocblas_client_diagonally_dominant_triangular_matrix,

} rocblas_check_matrix_type;

template <typename U, typename T>
void rocblas_init_matrix(rocblas_check_matrix_type matrix_type,
                         const char                uplo,
                         T                         rand_gen(),
                         U&                        hA)
{
    for(int64_t batch_index = 0; batch_index < hA.batch_count(); ++batch_index)
    {
        auto*   A   = hA[batch_index];
        int64_t M   = hA.m();
        int64_t N   = hA.n();
        int64_t lda = hA.lda();
        if(matrix_type == rocblas_client_general_matrix)
        {
#ifdef _OPENMP
#pragma omp parallel for
#endif
            for(size_t j = 0; j < N; ++j)
                for(size_t i = 0; i < M; ++i)
                    A[i + j * lda] = rand_gen();
        }
    }
}

template <typename U, typename T>
void rocblas_init_matrix_alternating_sign(rocblas_check_matrix_type matrix_type,
                                          const char                uplo,
                                          T                         rand_gen(),
                                          U&                        hA)
{
    for(int64_t batch_index = 0; batch_index < hA.batch_count(); ++batch_index)
    {
        auto* A   = hA[batch_index];
        auto  M   = hA.m();
        auto  N   = hA.n();
        auto  lda = hA.lda();

        if(matrix_type == rocblas_client_general_matrix)
        {
#ifdef _OPENMP
#pragma omp parallel for
#endif
            for(size_t i = 0; i < M; ++i)
                for(size_t j = 0; j < N; ++j)
                {
                    auto value     = rand_gen();
                    A[i + j * lda] = (i ^ j) & 1 ? T(value) : T(negate(value));
                }
        }
    }
}

/* ============================================================================================ */
/*! \brief  matrix/vector initialization: */
// for vector x (M=1, N=lengthX, lda=incx);
// for complex number, the real/imag part would be initialized with the same value

// Initialize vector with random values
template <typename T>
inline void rocblas_init(
    host_vector<T>& A, size_t M, size_t N, size_t lda, rocblas_stride stride = 0, size_t batch_count = 1)
{
    for(size_t i_batch = 0; i_batch < batch_count; i_batch++)
        for(size_t i = 0; i < M; ++i)
            for(size_t j = 0; j < N; ++j)
                A[i + j * lda + i_batch * stride] = random_generator<T>();
}

template <typename T>
inline void rocblas_init_sin(
    host_vector<T>& A, size_t M, size_t N, size_t lda, rocblas_stride stride = 0, size_t batch_count = 1)
{
    for(size_t i_batch = 0; i_batch < batch_count; i_batch++)
        for(size_t i = 0; i < M; ++i)
            for(size_t j = 0; j < N; ++j)
                A[i + j * lda + i_batch * stride] = T(sin(i + j * lda + i_batch * stride));
}

template <typename T, typename U>
void rocblas_init_matrix_trig(rocblas_check_matrix_type matrix_type,
                              const char                uplo,
                              U&                        hA,
                              bool                      seedReset = false)
{
    for(int64_t batch_index = 0; batch_index < hA.batch_count(); ++batch_index)
    {
        auto* A   = hA[batch_index];
        auto  M   = hA.m();
        auto  N   = hA.n();
        auto  lda = hA.lda();

        if(matrix_type == rocblas_client_general_matrix)
        {
#ifdef _OPENMP
#pragma omp parallel for
#endif
            for(size_t i = 0; i < M; ++i)
                for(size_t j = 0; j < N; ++j)
                    A[i + j * lda] = T(seedReset ? cos(i + j * lda) : sin(i + j * lda));
        }
    }
}

/* ============================================================================================ */
/*! \brief  Initialize an array with random data, with NaN where appropriate */

/*! \brief For testing purposes, copy one matrix into another with different leading dimensions  */
template <typename T, typename U>
void copy_matrix_with_different_leading_dimensions(T& hB, U& hC)
{
    rocblas_int M           = hB.m();
    rocblas_int N           = hB.n();
    size_t      ldb         = hB.lda();
    size_t      ldc         = hC.lda();
    rocblas_int batch_count = hB.batch_count();
    for(int b = 0; b < batch_count; b++)
    {
        auto* B = hB[b];
        auto* C = hC[b];
        for(int i = 0; i < M; i++)
            for(int j = 0; j < N; j++)
                C[i + j * ldc] = B[i + j * ldb];
    }
}

template <typename T>
inline void rocblas_init_nan(T* A, size_t N)
{
    for(size_t i = 0; i < N; ++i)
        A[i] = T(rocblas_nan_rng());
}

template <typename T>
void rocblas_init_nan(
    T* A, size_t M, size_t N, size_t lda, size_t stride = 0, size_t batch_count = 1)
{
    for(size_t i_batch = 0; i_batch < batch_count; i_batch++)
        for(size_t i = 0; i < M; ++i)
            for(size_t j = 0; j < N; ++j)
                A[i + j * lda + i_batch * stride] = T(rocblas_nan_rng());
}

template <typename T>
inline void rocblas_init_nan(
    host_vector<T>& A, size_t M, size_t N, size_t lda, rocblas_stride stride = 0, size_t batch_count = 1)
{
    for(size_t i_batch = 0; i_batch < batch_count; i_batch++)
        for(size_t i = 0; i < M; ++i)
            for(size_t j = 0; j < N; ++j)
                A[i + j * lda + i_batch * stride] = T(rocblas_nan_rng());
}

/* ============================================================================================ */
/*! \brief  Packs strided_batched matricies into groups of 4 in N */

template <typename T>
inline void rocblas_packInt8(
    std::vector<T>& A, size_t M, size_t N, size_t batch_count, size_t lda, rocblas_stride stride_a)
{
    if(N % 4 != 0)
        std::cerr << "ERROR: dimension must be a multiple of 4 in order to pack" << std::endl;

    std::vector<T> temp(A);
    for(size_t count = 0; count < batch_count; count++)
        for(size_t colBase = 0; colBase < N; colBase += 4)
            for(size_t row = 0; row < lda; row++)
                for(size_t colOffset = 0; colOffset < 4; colOffset++)
                    A[(colBase * lda + 4 * row) + colOffset + (stride_a * count)]
                        = temp[(colBase + colOffset) * lda + row + (stride_a * count)];
}

/* ============================================================================================ */
/*! \brief  Packs matricies into groups of 4 in N */
template <typename T>
inline void rocblas_packInt8(std::vector<T>& A, size_t M, size_t N, size_t lda)
{
    /* Assumes original matrix provided in column major order, where N is a multiple of 4

        ---------- N ----------
   |  | 00 05 10 15 20 25 30 35      |00 05 10 15|20 25 30 35|
   |  | 01 06 11 16 21 26 31 36      |01 06 11 16|21 26 31 36|
   l  M 02 07 12 17 22 27 32 37  --> |02 07 12 17|22 27 32 37|
   d  | 03 08 13 18 23 28 33 38      |03 08 13 18|23 28 33 38|
   a  | 04 09 14 19 24 29 34 39      |04 09 14 19|24 29 34 39|
   |    ** ** ** ** ** ** ** **      |** ** ** **|** ** ** **|
   |    ** ** ** ** ** ** ** **      |** ** ** **|** ** ** **|

     Input :  00 01 02 03 04 ** ** 05   ...  38 39 ** **
     Output:  00 05 10 15 01 06 11 16   ...  ** ** ** **

   */

    //  call general code with batch_count = 1 and stride_a = 0
    rocblas_packInt8(A, M, N, 1, lda, 0);
}

/* \brief floating point counts of GEMM */
template <typename T>
constexpr double gemm_gflop_count(rocblas_int m, rocblas_int n, rocblas_int k)
{
    return (2.0 * m * n * k) / 1e9;
}

template <>
constexpr double
    gemm_gflop_count<rocblas_float_complex>(rocblas_int m, rocblas_int n, rocblas_int k)
{
    return (8.0 * m * n * k) / 1e9;
}

template <>
constexpr double
    gemm_gflop_count<rocblas_double_complex>(rocblas_int m, rocblas_int n, rocblas_int k)
{
    return (8.0 * m * n * k) / 1e9;
}

typedef enum rocblas_initialization_
{
    rocblas_initialization_random_int    = 111,
    rocblas_initialization_random_narrow = 222,
    rocblas_initialization_random_broad  = 333,
    rocblas_initialization_random_full   = 444,
    rocblas_initialization_trig_float    = 555,
    rocblas_initialization_hpl           = 666,
    rocblas_initialization_const         = 777,
    rocblas_initialization_file          = 888,
} rocblas_initialization;

/* ============================================================================================ */
/*  Convert rocblas constants to lapack char. */

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

constexpr auto rocblas_initialization2string(rocblas_initialization init)
{
    switch(init)
    {
    case rocblas_initialization_random_int:
        return "rand_int";
    case rocblas_initialization_random_narrow:
        return "rand_narrow";
    case rocblas_initialization_random_broad:
        return "rand_broad";
    case rocblas_initialization_random_full:
        return "rand_full";
    case rocblas_initialization_trig_float:
        return "trig_float";
    case rocblas_initialization_hpl:
        return "hpl";
    case rocblas_initialization_const:
        return "const";
    case rocblas_initialization_file:
        return "file";
    default:
        return "invalid";
    }
}

/* ============================================================================================ */
/*  Convert lapack char constants to rocblas type. */

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

inline rocblas_initialization string2rocblas_initialization(const std::string& value)
{
    // clang-format off
    return
        value == "rand_int"   ? rocblas_initialization_random_int :
        value == "rand_narrow" ? rocblas_initialization_random_narrow:
        value == "rand_broad" ? rocblas_initialization_random_broad:
        value == "rand_full" ? rocblas_initialization_random_full:
        value == "trig_float" ? rocblas_initialization_trig_float :
        value == "hpl"        ? rocblas_initialization_hpl        :
        value == "const"        ? rocblas_initialization_const        :
        value == "file"        ? rocblas_initialization_file      :
        
        static_cast<rocblas_initialization>(-1);
    // clang-format on
}

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

class Barrier {
public:
    explicit Barrier(std::size_t count = 0) : 
      m_count(count+1), 
      m_start(0),
      m_threshold(count+1) 
    {
        for(int i = 0; i<m_count-1; i++)
            m_ready.push_back(false);

    }

    void init(std::size_t count)
    {
        m_ready.clear();
        m_count = count+1;
        m_start = 0;
        m_threshold = count+1;
        for(int i = 0; i<m_threshold-1; i++)
            m_ready.push_back(false);
    }

    void wait(int device = 0) 
    {
        std::unique_lock<std::mutex> lLock{m_mutex};
        auto start = m_start;
        if (!--m_count) {
            m_start++;
            m_count = m_threshold;
            m_cv.notify_all();
        } else {
            m_ready[device] = true;
            m_cv.wait(lLock, [this, start] { return start != m_start; });
        }
    }

    void wait_to_trigger() 
    {
        bool loop = true;
        while(loop)
        {
            bool ready = true;
            for(int i = 0; i<m_ready.size(); i++)
                if(m_ready[i]==false)
                    ready = false;
            asm("");
            loop = !ready;
        }

        wait();
    }

private:
    std::mutex m_mutex;
    std::condition_variable m_cv;
    std::size_t m_count;
    std::size_t m_start;
    std::size_t m_threshold;
    std::vector<bool> m_ready;
};

struct Arguments
{
    rocblas_int M;
    rocblas_int N;
    rocblas_int K;

    rocblas_int lda;
    rocblas_int ldb;
    rocblas_int ldc;
    rocblas_int ldd;

    rocblas_datatype a_type;
    rocblas_datatype b_type;
    rocblas_datatype c_type;
    rocblas_datatype d_type;
    rocblas_datatype compute_type;

    rocblas_int incx;
    rocblas_int incy;
    rocblas_int incd;
    rocblas_int incb;

    double alpha;
    double alphai;
    double beta;
    double betai;

    char transA;
    char transB;
    char side;
    char uplo;
    char diag;

    rocblas_int batch_count;

    rocblas_stride stride_a; //  stride_a > transA == 'N' ? lda * K : lda * M
    rocblas_stride stride_b; //  stride_b > transB == 'N' ? ldb * N : ldb * K
    rocblas_stride stride_c; //  stride_c > ldc * N
    rocblas_stride stride_d; //  stride_d > ldd * N

    double initVal;

    rocblas_int norm_check;
    rocblas_int unit_check;
    rocblas_int timing;
    rocblas_int iters;
    rocblas_int reinit_c;
    rocblas_int flush_gpu_cache;
    rocblas_int c_equals_d;
    rocblas_int time_each_iter;
    rocblas_int tensile_timing;

    uint32_t algo;
    int32_t  solution_index;
    uint32_t flags;

    char function[64];
    char name[64];
    char category[32];

    rocblas_initialization initialization;

    // Validate input format.
    // rocblas_gentest.py is expected to conform to this format.
    // rocblas_gentest.py uses rocblas_common.yaml to generate this format.
    static void validate(std::istream& ifs)
    {
        auto error = [](auto name) {
            std::cerr << "Arguments field " << name << " does not match format.\n\n"
                      << "Fatal error: Binary test data does match input format.\n"
                         "Ensure that rocblas_arguments.hpp and rocblas_common.yaml\n"
                         "define exactly the same Arguments, that rocblas_gentest.py\n"
                         "generates the data correctly, and that endianness is the same.\n";
            abort();
        };

        char      header[8] {}, trailer[8] {};
        Arguments arg {};
        ifs.read(header, sizeof(header));
        ifs >> arg;
        ifs.read(trailer, sizeof(trailer));

        if(strcmp(header, "rocBLAS"))
            error("header");
        else if(strcmp(trailer, "ROCblas"))
            error("trailer");

        auto check_func = [&, sig = (unsigned char)0](const auto& elem, auto name) mutable {
            static_assert(sizeof(elem) <= 255,
                          "One of the fields of Arguments is too large (> 255 bytes)");
            for(unsigned char i = 0; i < sizeof(elem); ++i)
                if(reinterpret_cast<const unsigned char*>(&elem)[i] ^ sig ^ i)
                    error(name);
            sig += 89;
        };

#define ROCBLAS_FORMAT_CHECK(x) check_func(arg.x, #x)

        // Order is important
        ROCBLAS_FORMAT_CHECK(M);
        ROCBLAS_FORMAT_CHECK(N);
        ROCBLAS_FORMAT_CHECK(K);
        ROCBLAS_FORMAT_CHECK(lda);
        ROCBLAS_FORMAT_CHECK(ldb);
        ROCBLAS_FORMAT_CHECK(ldc);
        ROCBLAS_FORMAT_CHECK(ldd);
        ROCBLAS_FORMAT_CHECK(a_type);
        ROCBLAS_FORMAT_CHECK(b_type);
        ROCBLAS_FORMAT_CHECK(c_type);
        ROCBLAS_FORMAT_CHECK(d_type);
        ROCBLAS_FORMAT_CHECK(initVal);
        ROCBLAS_FORMAT_CHECK(compute_type);
        ROCBLAS_FORMAT_CHECK(incx);
        ROCBLAS_FORMAT_CHECK(incy);
        ROCBLAS_FORMAT_CHECK(incd);
        ROCBLAS_FORMAT_CHECK(incb);
        ROCBLAS_FORMAT_CHECK(alpha);
        ROCBLAS_FORMAT_CHECK(alphai);
        ROCBLAS_FORMAT_CHECK(beta);
        ROCBLAS_FORMAT_CHECK(betai);
        ROCBLAS_FORMAT_CHECK(transA);
        ROCBLAS_FORMAT_CHECK(transB);
        ROCBLAS_FORMAT_CHECK(side);
        ROCBLAS_FORMAT_CHECK(uplo);
        ROCBLAS_FORMAT_CHECK(diag);
        ROCBLAS_FORMAT_CHECK(batch_count);
        ROCBLAS_FORMAT_CHECK(stride_a);
        ROCBLAS_FORMAT_CHECK(stride_b);
        ROCBLAS_FORMAT_CHECK(stride_c);
        ROCBLAS_FORMAT_CHECK(stride_d);
        ROCBLAS_FORMAT_CHECK(norm_check);
        ROCBLAS_FORMAT_CHECK(unit_check);
        ROCBLAS_FORMAT_CHECK(timing);
        ROCBLAS_FORMAT_CHECK(iters);
        ROCBLAS_FORMAT_CHECK(reinit_c);
        ROCBLAS_FORMAT_CHECK(flush_gpu_cache);
        ROCBLAS_FORMAT_CHECK(c_equals_d);
        ROCBLAS_FORMAT_CHECK(time_each_iter);
        ROCBLAS_FORMAT_CHECK(tensile_timing);
        ROCBLAS_FORMAT_CHECK(algo);
        ROCBLAS_FORMAT_CHECK(solution_index);
        ROCBLAS_FORMAT_CHECK(flags);
        ROCBLAS_FORMAT_CHECK(function);
        ROCBLAS_FORMAT_CHECK(name);
        ROCBLAS_FORMAT_CHECK(category);
        ROCBLAS_FORMAT_CHECK(initialization);
    }

    template <typename T>
    T get_alpha() const
    {
        return rocblas_isnan(alpha) || rocblas_isnan(alphai) ? T(0)
                                                             : convert_alpha_beta<T>(alpha, alphai);
    }

    template <typename T>
    T get_beta() const
    {
        return rocblas_isnan(beta) || rocblas_isnan(betai) ? T(0)
                                                           : convert_alpha_beta<T>(beta, betai);
    }

private:
    template <typename T, typename U, typename std::enable_if<!rocblas_is_complex<T>, int>::type = 0>
    static T convert_alpha_beta(U r, U i)
    {
        return T(r);
    }

    template <typename T, typename U, typename std::enable_if<+rocblas_is_complex<T>, int>::type = 0>
    static T convert_alpha_beta(U r, U i)
    {
        return T(r, i);
    }

    // Function to read Structures data from stream
    friend std::istream& operator>>(std::istream& str, Arguments& arg)
    {
        str.read(reinterpret_cast<char*>(&arg), sizeof(arg));
        return str;
    }

    // print_value is for formatting different data types

    // Default output
    template <typename T>
    static void print_value(std::ostream& str, const T& x)
    {
        str << x;
    }

    // Floating-point output
    static void print_value(std::ostream& str, double x)
    {
        if(std::isnan(x))
            str << ".nan";
        else if(std::isinf(x))
            str << (x < 0 ? "-.inf" : ".inf");
        else
        {
            char s[32];
            snprintf(s, sizeof(s) - 2, "%.17g", x);

            // If no decimal point or exponent, append .0
            char* end = s + strcspn(s, ".eE");
            if(!*end)
                strcat(end, ".0");
            str << s;
        }
    }

    // Character output
    static void print_value(std::ostream& str, char c)
    {
        char s[] {c, 0};
        str << std::quoted(s, '\'');
    }

    // bool output
    static void print_value(std::ostream& str, bool b)
    {
        str << (b ? "true" : "false");
    }

    // string output
    static void print_value(std::ostream& str, const char* s)
    {
        str << std::quoted(s);
    }

    // Function to print Arguments out to stream in YAML format
    // Google Tests uses this automatically to dump parameters
    friend std::ostream& operator<<(std::ostream& str, const Arguments& arg)
    {
        // delim starts as '{' opening brace and becomes ',' afterwards
        auto print = [&, delim = '{'](const char* name, auto x) mutable {
            str << delim << " " << name << ": ";
            print_value(str, x);
            delim = ',';
        };

        print("function", arg.function);
        print("a_type", rocblas_datatype2string(arg.a_type));
        print("b_type", rocblas_datatype2string(arg.b_type));
        print("c_type", rocblas_datatype2string(arg.c_type));
        print("d_type", rocblas_datatype2string(arg.d_type));
        print("compute_type", rocblas_datatype2string(arg.compute_type));
        print("transA", arg.transA);
        print("transB", arg.transB);
        print("M", arg.M);
        print("N", arg.N);
        print("K", arg.K);
        print("lda", arg.lda);
        print("ldb", arg.ldb);
        print("ldc", arg.ldc);
        print("ldd", arg.ldd);
        print("incx", arg.incx);
        print("incy", arg.incy);
        print("incd", arg.incd);
        print("incb", arg.incb);
        print("alpha", arg.alpha);
        print("alphai", arg.alphai);
        print("beta", arg.beta);
        print("betai", arg.betai);
        print("side", arg.side);
        print("uplo", arg.uplo);
        print("diag", arg.diag);
        print("batch_count", arg.batch_count);
        print("stride_a", arg.stride_a);
        print("stride_b", arg.stride_b);
        print("stride_c", arg.stride_c);
        print("stride_d", arg.stride_d);
        print("initVal", arg.initVal);
        print("algo", arg.algo);
        print("solution_index", arg.solution_index);
        print("flags", arg.flags);
        print("name", arg.name);
        print("category", arg.category);
        print("norm_check", arg.norm_check);
        print("unit_check", arg.unit_check);
        print("timing", arg.timing);
        print("iters", arg.iters);
        print("reinit_c", arg.reinit_c);
        print("flush_gpu_cache", arg.flush_gpu_cache);
        print("c_equals_d", arg.c_equals_d);
        print("time_each_iter", arg.time_each_iter);
        print("tensile_timing", arg.tensile_timing);
        print("initialization", arg.initialization);

        return str << " }\n";
    }
};

static_assert(std::is_standard_layout<Arguments> {},
              "Arguments is not a standard layout type, and thus is incompatible with C.");

static_assert(std::is_trivially_copyable<Arguments> {},
              "Arguments is not a trivially copyable type, and thus is incompatible with C.");

std::string function;
std::string precision;
std::string a_type;
std::string b_type;
std::string c_type;
std::string d_type;
std::string compute_type;
std::string initialization;
std::string a_file, b_file, c_file, o_file;
rocblas_int storeInitData;
rocblas_int storeOutputData;
rocblas_int device_id;
rocblas_int multi_device;
Barrier perfBarrier;
Barrier memBarrier;
Barrier memBarrier2;

template <typename T>
inline rocblas_stride align_stride(rocblas_stride stride)
{
    // hipMalloc aligns pointers on 256 byte boundaries (or a multiple of 256)
    // this function is to align stride*sizeof(T) on 256 byte boundaries
    size_t byte_alignment = 256;

    if(byte_alignment % sizeof(T) == 0)
    {
        size_t type_alignment = byte_alignment / sizeof(T);
        return ((stride - 1) / type_alignment + 1) * type_alignment;
    }
    else
    {
        return ((stride - 1) / byte_alignment + 1) * byte_alignment;
    }
}

//
// Forward declaration of the host matrix.
//
template <typename T>
class host_matrix;

//!
//! @brief pseudo-matrix subclass which uses device memory
//!
template <typename T>
class device_matrix : public d_vector<T>
{

public:
    //!
    //! @brief Disallow copying.
    //!
    device_matrix(const device_matrix&) = delete;

    //!
    //! @brief Disallow assigning
    //!
    device_matrix& operator=(const device_matrix&) = delete;

    //!
    //! @brief Constructor.
    //! @param m           The number of rows of the Matrix.
    //! @param n           The number of cols of the Matrix.
    //! @param lda         The leading dimension of the Matrix.
    //! @param HMM         HipManagedMemory Flag.
    //!
    explicit device_matrix(size_t m, size_t n, size_t lda, bool HMM = false)
        : d_vector<T>{n * lda, HMM}
        , m_m{m}
        , m_n{n}
        , m_lda{lda}
        , m_data{this->device_vector_setup()}
    {
    }

    //!
    //! @brief Destructor.
    //!
    ~device_matrix()
    {
        this->device_vector_teardown(m_data);
        m_data = nullptr;
    }

    //!
    //! @brief Returns the rows of the Matrix.
    //!
    size_t m() const
    {
        return this->m_m;
    }

    //!
    //! @brief Returns the cols of the Matrix.
    //!
    size_t n() const
    {
        return this->m_n;
    }

    //!
    //! @brief Returns the leading dimension of the Matrix.
    //!
    size_t lda() const
    {
        return this->m_lda;
    }

    //!
    //! @brief Returns the batch count (always 1).
    //!
    rocblas_int batch_count() const
    {
        return 1;
    }

    //!
    //! @brief Returns the stride (out of context, always 0)
    //!
    rocblas_stride stride() const
    {
        return 0;
    }

    //!
    //! @brief Decay into pointer wherever pointer is expected.
    //!
    operator T*()
    {
        return m_data;
    }

    //!
    //! @brief Decay into constant pointer wherever pointer is expected.
    //!
    operator const T*() const
    {
        return m_data;
    }

    //!
    //! @brief Transfer data from a host matrix.
    //! @param that The host matrix.
    //! @return the hip error.
    //!
    hipError_t transfer_from(const host_matrix<T>& that)
    {
        return hipMemcpy(m_data,
                         (const T*)that,
                         this->nmemb() * sizeof(T),
                         this->use_HMM ? hipMemcpyHostToHost : hipMemcpyHostToDevice);
    }

    hipError_t memcheck() const
    {
        return !this->nmemb() || m_data ? hipSuccess : hipErrorOutOfMemory;
    }

private:
    size_t m_m   = 0;
    size_t m_n   = 0;
    size_t m_lda = 0;
    T*     m_data{};
};

//
// Local declaration of the host strided batch matrix.
//
template <typename T>
class host_strided_batch_matrix;

//!
//! @brief Implementation of a strided batched matrix on device.
//!
template <typename T>
class device_strided_batch_matrix : public d_vector<T>
{
public:
    //!
    //! @brief Disallow copying.
    //!
    device_strided_batch_matrix(const device_strided_batch_matrix&) = delete;

    //!
    //! @brief Disallow assigning.
    //!
    device_strided_batch_matrix& operator=(const device_strided_batch_matrix&) = delete;

    //!
    //! @brief Constructor.
    //! @param m           The number of rows of the Matrix.
    //! @param n           The number of cols of the Matrix.
    //! @param lda         The leading dimension of the Matrix.
    //! @param stride The stride.
    //! @param batch_count The batch count.
    //! @param HMM         HipManagedMemory Flag.
    //!
    explicit device_strided_batch_matrix(size_t         m,
                                         size_t         n,
                                         size_t         lda,
                                         rocblas_stride stride,
                                         int64_t        batch_count,
                                         bool           HMM = false)
        : d_vector<T>(calculate_nmemb(n, lda, stride, batch_count), HMM)
        , m_m(m)
        , m_n(n)
        , m_lda(lda)
        , m_stride(stride)
        , m_batch_count(batch_count)
    {
        bool valid_parameters = calculate_nmemb(n, lda, stride, batch_count) > 0;
        if(valid_parameters)
        {
            this->m_data = this->device_vector_setup();
        }
    }

    //!
    //! @brief Destructor.
    //!
    ~device_strided_batch_matrix()
    {
        if(nullptr != this->m_data)
        {
            this->device_vector_teardown(this->m_data);
            this->m_data = nullptr;
        }
    }

    //!
    //! @brief Returns the data pointer.
    //!
    T* data()
    {
        return this->m_data;
    }

    //!
    //! @brief Returns the data pointer.
    //!
    const T* data() const
    {
        return this->m_data;
    }

    //!
    //! @brief Returns the rows of the Matrix.
    //!
    size_t m() const
    {
        return this->m_m;
    }

    //!
    //! @brief Returns the cols of the Matrix.
    //!
    size_t n() const
    {
        return this->m_n;
    }

    //!
    //! @brief Returns the leading dimension of the Matrix.
    //!
    size_t lda() const
    {
        return this->m_lda;
    }

    //!
    //! @brief Returns the batch count.
    //!
    int64_t batch_count() const
    {
        return this->m_batch_count;
    }

    //!
    //! @brief Returns the stride value.
    //!
    rocblas_stride stride() const
    {
        return this->m_stride;
    }

    //!
    //! @brief Returns pointer.
    //! @param batch_index The batch index.
    //! @return A mutable pointer to the batch_index'th matrix.
    //!
    T* operator[](int64_t batch_index)
    {
        return (this->m_stride >= 0)
                   ? this->m_data + batch_index * this->m_stride
                   : this->m_data + (batch_index + 1 - this->m_batch_count) * this->m_stride;
    }

    //!
    //! @brief Returns non-mutable pointer.
    //! @param batch_index The batch index.
    //! @return A non-mutable mutable pointer to the batch_index'th matrix.
    //!
    const T* operator[](int64_t batch_index) const
    {
        return (this->m_stride >= 0)
                   ? this->m_data + batch_index * this->m_stride
                   : this->m_data + (batch_index + 1 - this->m_batch_count) * this->m_stride;
    }

    //!
    //! @brief Cast operator.
    //! @remark Returns the pointer of the first matrix.
    //!
    operator T*()
    {
        return (*this)[0];
    }

    //!
    //! @brief Non-mutable cast operator.
    //! @remark Returns the non-mutable pointer of the first matrix.
    //!
    operator const T*() const
    {
        return (*this)[0];
    }

    //!
    //! @brief Tell whether resource allocation failed.
    //!
    explicit operator bool() const
    {
        return nullptr != this->m_data;
    }

    //!
    //! @brief Transfer data from a strided batched matrix on device.
    //! @param that That strided batched matrix on device.
    //! @return The hip error.
    //!
    hipError_t transfer_from(const host_strided_batch_matrix<T>& that)
    {
        return hipMemcpy(this->data(),
                         that.data(),
                         sizeof(T) * this->nmemb(),
                         this->use_HMM ? hipMemcpyHostToHost : hipMemcpyHostToDevice);
    }

    //!
    //! @brief Broadcast data from one matrix on host to each batch_count matrices.
    //! @param that That matrix on host.
    //! @return The hip error.
    //!
    hipError_t broadcast_one_matrix_from(const host_matrix<T>& that)
    {
        hipError_t status = hipSuccess;
        for(int64_t batch_index = 0; batch_index < m_batch_count; batch_index++)
        {
            status = hipMemcpy(this->data() + (batch_index * m_stride),
                               that.data(),
                               sizeof(T) * this->m_n * this->m_lda,
                               this->use_HMM ? hipMemcpyHostToHost : hipMemcpyHostToDevice);
            if(status != hipSuccess)
                break;
        }
        return status;
    }

    //!
    //! @brief Check if memory exists.
    //! @return hipSuccess if memory exists, hipErrorOutOfMemory otherwise.
    //!
    hipError_t memcheck() const
    {
        bool valid_parameters = calculate_nmemb(m_n, m_lda, m_stride, m_batch_count) > 0;

        if(*this || !valid_parameters)
            return hipSuccess;
        else
            return hipErrorOutOfMemory;
    }

private:
    size_t         m_m{};
    size_t         m_n{};
    size_t         m_lda{};
    rocblas_stride m_stride{};
    int64_t        m_batch_count{};
    T*             m_data{};

    static size_t calculate_nmemb(size_t n, size_t lda, rocblas_stride stride, int64_t batch_count)
    {
        return lda * n + size_t(batch_count - 1) * std::abs(stride);
    }
};

template <typename T>
struct host_matrix : std::vector<T, host_memory_allocator<T>>
{
    // Inherit constructors
    using std::vector<T, host_memory_allocator<T>>::vector;

    //!
    //! @brief Constructor.
    //!
    host_matrix(size_t m, size_t n, size_t lda)
        : std::vector<T, host_memory_allocator<T>>(n * lda)
        , m_m(m)
        , m_n(n)
        , m_lda(lda)
    {
    }

    //!
    //! @brief Copy constructor from host_matrix of other types convertible to T
    //!
    template <typename U, std::enable_if_t<std::is_convertible<U, T>{}, int> = 0>
    host_matrix(const host_matrix<U>& x)
        : std::vector<T, host_memory_allocator<T>>(x.size())
        , m_m(x.size())
        , m_n(1)
        , m_lda(1)
    {
        for(size_t i = 0; i < m_m; ++i)
            (*this)[i] = x[i];
    }

    //!
    //! @brief Decay into pointer wherever pointer is expected
    //!
    operator T*()
    {
        return this->data();
    }

    //!
    //! @brief Decay into constant pointer wherever constant pointer is expected
    //!
    operator const T*() const
    {
        return this->data();
    }

    //!
    //! @brief Transfer from a device matrix.
    //! @param  that That device matrix.
    //! @return the hip error.
    //!
    hipError_t transfer_from(const device_matrix<T>& that)
    {
        hipError_t hip_err;

        if(that.use_HMM && hipSuccess != (hip_err = hipDeviceSynchronize()))
            return hip_err;

        return hipMemcpy(*this,
                         that,
                         sizeof(T) * this->size(),
                         that.use_HMM ? hipMemcpyHostToHost : hipMemcpyDeviceToHost);
    }

    //!
    //! @brief Transfer only the first matrix from a device_strided_batch matrix.
    //! @param  that That device_strided_batch matrix.
    //! @return the hip error.
    //!
    hipError_t transfer_one_matrix_from(const device_strided_batch_matrix<T>& that)
    {
        hipError_t hip_err;

        if(that.use_HMM && hipSuccess != (hip_err = hipDeviceSynchronize()))
            return hip_err;

        return hipMemcpy(*this,
                         that,
                         sizeof(T) * this->size(),
                         that.use_HMM ? hipMemcpyHostToHost : hipMemcpyDeviceToHost);
    }

    //!
    //! @brief Returns the rows of the Matrix.
    //!
    size_t m() const
    {
        return this->m_m;
    }

    //!
    //! @brief Returns the cols of the Matrix.
    //!
    size_t n() const
    {
        return this->m_n;
    }

    //!
    //! @brief Returns the leading dimension of the Matrix.
    //!
    size_t lda() const
    {
        return this->m_lda;
    }

    //!
    //! @brief Returns the batch count (always 1).
    //!
    static constexpr rocblas_int batch_count()
    {
        return 1;
    }

    //!
    //! @brief Returns the stride (out of context, always 0)
    //!
    static constexpr rocblas_stride stride()
    {
        return 0;
    }

    //!
    //! @brief Random access to the Matrices.
    //! @param batch_index the batch index.
    //! @return The mutable pointer.
    //!
    T* operator[](int64_t batch_index)
    {
        return this->data();
    }

    //!
    //! @brief Constant random access to the Matrices.
    //! @param batch_index the batch index.
    //! @return The non-mutable pointer.
    //!
    const T* operator[](int64_t batch_index) const
    {
        return this->data();
    }

    //!
    //! @brief Check if memory exists (out of context, always hipSuccess)
    //!
    static constexpr hipError_t memcheck()
    {
        return hipSuccess;
    }

private:
    size_t m_m   = 0;
    size_t m_n   = 0;
    size_t m_lda = 0;
};

//!
//! @brief Implementation of a host strided batched matrix.
//!
template <typename T>
class host_strided_batch_matrix
{
public:
    //!
    //! @brief Disallow copying.
    //!
    host_strided_batch_matrix(const host_strided_batch_matrix&) = delete;

    //!
    //! @brief Disallow assigning.
    //!
    host_strided_batch_matrix& operator=(const host_strided_batch_matrix&) = delete;

    //!
    //! @brief Constructor.
    //! @param m           The number of rows of the Matrix.
    //! @param n           The number of cols of the Matrix.
    //! @param lda         The leading dimension of the Matrix.
    //! @param stride The stride.
    //! @param batch_count The batch count.
    //!
    explicit host_strided_batch_matrix(
        size_t m, size_t n, size_t lda, rocblas_stride stride, int64_t batch_count)
        : m_m(m)
        , m_n(n)
        , m_lda(lda)
        , m_stride(stride)
        , m_batch_count(batch_count)
        , m_nmemb(calculate_nmemb(n, lda, stride, batch_count))
    {
        bool valid_parameters = this->m_nmemb > 0;
        if(valid_parameters)
        {
            this->m_data = (T*)host_calloc_throw(this->m_nmemb, sizeof(T));
        }
    }

    //!
    //! @brief Destructor.
    //!
    ~host_strided_batch_matrix()
    {
        if(nullptr != this->m_data)
        {
            free(this->m_data);
            this->m_data = nullptr;
        }
    }

    //!
    //! @brief Returns the data pointer.
    //!
    T* data()
    {
        return this->m_data;
    }

    //!
    //! @brief Returns the data pointer.
    //!
    const T* data() const
    {
        return this->m_data;
    }

    //!
    //! @brief Returns the rows of the Matrix.
    //!
    size_t m() const
    {
        return this->m_m;
    }

    //!
    //! @brief Returns the cols of the Matrix.
    //!
    size_t n() const
    {
        return this->m_n;
    }

    //!
    //! @brief Returns the leading dimension of the Matrix.
    //!
    size_t lda() const
    {
        return this->m_lda;
    }

    //!
    //! @brief Returns the batch count.
    //!
    int64_t batch_count() const
    {
        return this->m_batch_count;
    }

    //!
    //! @brief Returns the stride.
    //!
    rocblas_stride stride() const
    {
        return this->m_stride;
    }

    //!
    //! @brief Returns nmemb.
    //!
    size_t nmemb() const
    {
        return this->m_nmemb;
    }

    //!
    //! @brief Returns pointer.
    //! @param batch_index The batch index.
    //! @return A mutable pointer to the batch_index'th matrix.
    //!
    T* operator[](int64_t batch_index)
    {

        return (this->m_stride >= 0)
                   ? this->m_data + this->m_stride * batch_index
                   : this->m_data + (batch_index + 1 - this->m_batch_count) * this->m_stride;
    }

    //!
    //! @brief Returns non-mutable pointer.
    //! @param batch_index The batch index.
    //! @return A non-mutable mutable pointer to the batch_index'th matrix.
    //!
    const T* operator[](int64_t batch_index) const
    {

        return (this->m_stride >= 0)
                   ? this->m_data + this->m_stride * batch_index
                   : this->m_data + (batch_index + 1 - this->m_batch_count) * this->m_stride;
    }

    //!
    //! @brief Cast operator.
    //! @remark Returns the pointer of the first matrix.
    //!
    operator T*()
    {
        return (*this)[0];
    }

    //!
    //! @brief Non-mutable cast operator.
    //! @remark Returns the non-mutable pointer of the first matrix.
    //!
    operator const T*() const
    {
        return (*this)[0];
    }

    //!
    //! @brief Tell whether resource allocation failed.
    //!
    explicit operator bool() const
    {
        return nullptr != this->m_data;
    }

    //!
    //! @brief Copy data from a strided batched matrix on host.
    //! @param that That strided batched matrix on host.
    //! @return true if successful, false otherwise.
    //!
    bool copy_from(const host_strided_batch_matrix& that)
    {
        if(that.m() == this->m_m && that.n() == this->m_n && that.lda() == this->m_lda
           && that.stride() == this->m_stride && that.batch_count() == this->m_batch_count)
        {
            memcpy(this->data(), that.data(), sizeof(T) * this->m_nmemb);
            return true;
        }
        else
        {
            return false;
        }
    }

    //!
    //! @brief Transfer data from a strided batched matrix on device.
    //! @param that That strided batched matrix on device.
    //! @return The hip error.
    //!
    hipError_t transfer_from(const device_strided_batch_matrix<T>& that)
    {
        hipError_t hip_err;

        if(that.use_HMM && hipSuccess != (hip_err = hipDeviceSynchronize()))
            return hip_err;

        return hipMemcpy(this->m_data,
                         that.data(),
                         sizeof(T) * this->m_nmemb,
                         that.use_HMM ? hipMemcpyHostToHost : hipMemcpyDeviceToHost);
    }

    //!
    //! @brief Check if memory exists.
    //! @return hipSuccess if memory exists, hipErrorOutOfMemory otherwise.
    //!
    hipError_t memcheck() const
    {
        return ((bool)*this) ? hipSuccess : hipErrorOutOfMemory;
    }

private:
    size_t         m_m{};
    size_t         m_n{};
    size_t         m_lda{};
    rocblas_stride m_stride{};
    int64_t        m_batch_count{};
    size_t         m_nmemb{};
    T*             m_data{};

    static size_t calculate_nmemb(size_t n, size_t lda, rocblas_stride stride, int64_t batch_count)
    {
        return lda * n + size_t(batch_count - 1) * std::abs(stride);
    }
};

//!
//! @brief Overload output operator.
//! @param os The ostream.
//! @param that That host strided batch matrix.
//!
template <typename T>
internal_ostream& operator<<(internal_ostream&           os,
                                     const host_strided_batch_matrix<T>& that)
{
    auto m           = that.m();
    auto n           = that.n();
    auto lda         = that.lda();
    auto batch_count = that.batch_count();

    for(int64_t batch_index = 0; batch_index < batch_count; ++batch_index)
    {
        auto batch_data = that[batch_index];
        os << "[" << batch_index << "]" << std::endl;
        for(size_t i = 0; i < n; ++i)
        {
            for(size_t j = 0; j < m; ++j)
                os << ", " << batch_data[j + i * lda];
        }
        os << std::endl;
    }

    return os;
}

void readArgs(int argc, char* argv[], Arguments& arg)
{
    boost::program_options::options_description desc("rocblas-bench command line options");
    desc.add_options()
        // clang-format off
    ("sizem,m",
        po::value<rocblas_int>(&arg.M)->default_value(128),
        "Specific matrix size: sizem is only applicable to BLAS-2 & BLAS-3: the number of "
        "rows or columns in matrix.")

    ("sizen,n",
        po::value<rocblas_int>(&arg.N)->default_value(128),
        "Specific matrix/vector size: BLAS-1: the length of the vector. BLAS-2 & "
        "BLAS-3: the number of rows or columns in matrix")

    ("sizek,k",
        po::value<rocblas_int>(&arg.K)->default_value(128),
        "Specific matrix size:sizek is only applicable to BLAS-3: the number of columns in "
        "A and rows in B.")

    ("lda",
        po::value<rocblas_int>(&arg.lda)->default_value(128),
        "On entry, LDA specifies the first dimension of A as declared"
           "in the calling (sub) program. When  TRANSA = 'N' or 'n' then"
           "LDA must be at least  max( 1, m ), otherwise  LDA must be at"
           "least  max( 1, k )")

    ("ldb",
        po::value<rocblas_int>(&arg.ldb)->default_value(128),
        "On entry, LDB specifies the first dimension of B as declared"
           "in the calling (sub) program. When  TRANSB = 'N' or 'n' then"
           "LDB must be at least  max( 1, k ), otherwise  LDB must be at"
           "least  max( 1, n ).")

    ("ldc",
        po::value<rocblas_int>(&arg.ldc)->default_value(128),
        "On entry, LDC specifies the first dimension of C as declared"
           "in  the  calling  (sub)  program.   LDC  must  be  at  least"
           "max( 1, m ).")

    ("ldd",
        po::value<rocblas_int>(&arg.ldd)->default_value(128),
        "On entry, LDD specifies the first dimension of D as desired"
           "in  the  calling  (sub)  program.   LDD  must  be  at  least"
           "max( 1, m ).")

    ("stride_a",
        po::value<rocblas_stride>(&arg.stride_a)->default_value(128*128),
        "Specific stride of strided_batched matrix A, is only applicable to strided batched"
        "BLAS-2 and BLAS-3: second dimension * leading dimension.")

    ("stride_b",
        po::value<rocblas_stride>(&arg.stride_b)->default_value(128*128),
        "Specific stride of strided_batched matrix B, is only applicable to strided batched"
        "BLAS-2 and BLAS-3: second dimension * leading dimension.")

    ("stride_c",
        po::value<rocblas_stride>(&arg.stride_c)->default_value(128*128),
        "Specific stride of strided_batched matrix C, is only applicable to strided batched"
        "BLAS-2 and BLAS-3: second dimension * leading dimension.")

    ("stride_d",
        po::value<rocblas_stride>(&arg.stride_d)->default_value(128*128),
        "Specific stride of strided_batched matrix D, is only applicable to strided batched"
        "BLAS_EX: second dimension * leading dimension.")

    ("alpha",
        po::value<double>(&arg.alpha)->default_value(1.0), "specifies the scalar alpha")

    ("beta",
        po::value<double>(&arg.beta)->default_value(0.0), "specifies the scalar beta")
        
    ("initVal",
    po::value<double>(&arg.initVal)->default_value(1.0), "specifies the const value to be used with const initialization")

    ("function,f",
        po::value<std::string>(&function)->default_value("gemm"),
        "GEMM function to test. (gemm, gemm_strided_batched and gemm_ex")

    ("precision,r",
        po::value<std::string>(&precision)->default_value("f32_r"), "Precision. "
        "Options: s,d,f16_r,bf16_r,f32_r,f64_r")

    ("a_type",
        po::value<std::string>(&a_type), "Precision of matrix A. "
        "Options: s,d,f32_r,f64_r")

    ("b_type",
        po::value<std::string>(&b_type), "Precision of matrix B. "
        "Options: s,d,f32_r,f64_r")

    ("c_type",
        po::value<std::string>(&c_type), "Precision of matrix C. "
        "Options: s,d,f32_r,f64_r")

    ("d_type",
        po::value<std::string>(&d_type), "Precision of matrix D. "
        "Options: s,d,f32_r,f64_r")

    ("compute_type",
        po::value<std::string>(&compute_type), "Precision of computation. "
        "Options: s,d,b16_r,f32_r,f64_r")

    ("initialization",
        po::value<std::string>(&initialization)->default_value("rand_int"),
        "Intialize with random numbers, trig functions sin and cos, hpl-like input, or by loading data from a bin file"
        "Options: rand_int, rand_narrow, rand_broad, rand_full, trig_float, hpl, const, file")

    ("storeInitData,s",
        po::value<rocblas_int>(&storeInitData)->default_value(0),
        "Dump initialization data in to bin files? Note: Storing is not done when loading from bin files. "
        "Please specify file names using --x_file flags 0 = No, 1 = Yes (default: No)")

    ("storeOutputData,o",
        po::value<rocblas_int>(&storeOutputData)->default_value(0),
        "Dump results matrix in to bin files?"
        "Please specify file names using --o_file flag 0 = No, 1 = Yes (default: No) "
        "Note that multiple iterations will change results unless reinit_c flag is specified")

    ("a_file",
        po::value<std::string>(&a_file)->default_value(""), "Bin file storing matrix A. "
        "Options: text.bin")

    ("b_file",
        po::value<std::string>(&b_file)->default_value(""), "Bin file storing matrix B. "
        "Options: text.bin")

    ("c_file",
        po::value<std::string>(&c_file)->default_value(""), "Bin file storing matrix C. "
        "Options: text.bin")

    ("o_file",
        po::value<std::string>(&o_file)->default_value(""), "Bin file storing result matrix. "
        "Options: text.bin")

    ("transposeA",
        po::value<char>(&arg.transA)->default_value('N'),
        "N = no transpose, T = transpose, C = conjugate transpose")

    ("transposeB",
        po::value<char>(&arg.transB)->default_value('N'),
        "N = no transpose, T = transpose, C = conjugate transpose")

    ("batch",
        po::value<rocblas_int>(&arg.batch_count)->default_value(1),
        "Number of matrices. Only applicable to batched routines") // xtrsm xtrsm_ex xtrmm xgemm

    ("verify,v",
        po::value<rocblas_int>(&arg.norm_check)->default_value(0),
        "Validate GPU results with CPU? 0 = No, 1 = Yes (default: No)")

    ("unit_check,u",
        po::value<rocblas_int>(&arg.unit_check)->default_value(0),
        "Unit Check? 0 = No, 1 = Yes (default: No)")

    ("iters,i",
        po::value<rocblas_int>(&arg.iters)->default_value(10),
        "Iterations to run inside timing loop")
    
    ("reinit_c",
        po::value<rocblas_int>(&arg.reinit_c)->default_value(0),
        "Reinitialize C between iterations? 0 = No, 1 = Yes (default: No)")

    ("flush_gpu_cache",
        po::value<rocblas_int>(&arg.flush_gpu_cache)->default_value(0),
        "Flush GPU L2 cache between iterations? 0 = No, 1 = Yes (default: No)")

    ("c_equals_d",
        po::value<rocblas_int>(&arg.c_equals_d)->default_value(1),
        "is C equal to D? 0 = No, 1 = Yes (default: Yes)")

    ("time_each_iter",
        po::value<rocblas_int>(&arg.time_each_iter)->default_value(0),
        "Explicitly time each iteration? 0 = No, 1 = Yes (default: No)")

    ("tensile_timing",
        po::value<rocblas_int>(&arg.tensile_timing)->default_value(0),
        "Get kernel timing from Tensile? This sends hipEvents directly to the kernel call,"
        " eliminating overhead that may be seen for smaller launches. "
        "Will use this timing to calculate performance when enabled.\n"
         "Options: 0 = No, 1 = Yes (default: No)")

    ("algo",
        po::value<uint32_t>(&arg.algo)->default_value(0),
        "extended precision gemm algorithm")

    ("solution_index",
        po::value<int32_t>(&arg.solution_index)->default_value(0),
        "extended precision gemm solution index")

    ("flags",
        po::value<uint32_t>(&arg.flags)->default_value(10),
        "extended precision gemm flags")

    ("device",
        po::value<rocblas_int>(&device_id)->default_value(0),
        "Set default device to be used for subsequent program runs (default: 0)")

    ("multi_device",
        po::value<rocblas_int>(&multi_device)->default_value(1),
        "This flag is used to specify how many devices to launch work on simultaneously (default: 1)"
        "The first x amount of devices will be used. Multiple threads will sync after setup for each device."
        "Then a rocblas call will be deployed to each device simultaneously and the longest timing duration will be pulled."
        "Each device will run iters iterations, and total performance will be calculated as combined iterations"
        "Flag cannot be combined with time_each_iter")

    ("help,h", "produces this help message");

    // clang-format on

    po::variables_map vm;
    store(parse_command_line(argc, argv, desc), vm);
    notify(vm);

    if(vm.count("help"))
    {
        std::cout << desc << std::endl;
        exit(1);
    }

    if(vm["reinit_c"].defaulted() && storeOutputData)
        arg.reinit_c = 1;

    std::transform(precision.begin(), precision.end(), precision.begin(), ::tolower);
    auto prec = string2rocblas_datatype(precision);
    if(prec == static_cast<rocblas_datatype>(-1))
        throw std::invalid_argument("Invalid value for --precision " + precision);

    if(a_type == "")
        a_type = precision;
    arg.a_type = string2rocblas_datatype(a_type);
    if(arg.a_type == static_cast<rocblas_datatype>(-1))
        throw std::invalid_argument("Invalid value for --a_type " + a_type);

    if(b_type == "")
        b_type = precision;
    arg.b_type = string2rocblas_datatype(b_type);
    if(arg.b_type == static_cast<rocblas_datatype>(-1))
        throw std::invalid_argument("Invalid value for --b_type " + b_type);

    if(c_type == "")
        c_type = precision;
    arg.c_type = string2rocblas_datatype(c_type);
    if(arg.c_type == static_cast<rocblas_datatype>(-1))
        throw std::invalid_argument("Invalid value for --c_type " + c_type);

    if(d_type == "")
        d_type = precision;
    arg.d_type = string2rocblas_datatype(d_type);
    if(arg.d_type == static_cast<rocblas_datatype>(-1))
        throw std::invalid_argument("Invalid value for --d_type " + d_type);

    if(compute_type == "")
        compute_type = precision;
    arg.compute_type = string2rocblas_datatype(compute_type);
    if(arg.compute_type == static_cast<rocblas_datatype>(-1))
        throw std::invalid_argument("Invalid value for --compute_type " + compute_type);

    arg.initialization = string2rocblas_initialization(initialization);
    if(arg.initialization == static_cast<rocblas_initialization>(-1))
        throw std::invalid_argument("Invalid value for --initialization " + initialization);

    if(arg.M < 0)
        throw std::invalid_argument("Invalid value for -m " + std::to_string(arg.M));
    if(arg.N < 0)
        throw std::invalid_argument("Invalid value for -n " + std::to_string(arg.N));
    if(arg.K < 0)
        throw std::invalid_argument("Invalid value for -k " + std::to_string(arg.K));

    if(arg.initialization
       == rocblas_initialization_file) //check for files if initialization is file
    {
        if(!std::ifstream(a_file))
            throw std::invalid_argument("Invalid value for --a_file " + a_file);
        if(!std::ifstream(b_file))
            throw std::invalid_argument("Invalid value for --b_file " + b_file);
        if(!std::ifstream(c_file))
            throw std::invalid_argument("Invalid value for --c_file " + c_file);
    }
    else if(arg.initialization == rocblas_initialization_const && std::isnan(arg.initVal))
    {
        throw std::invalid_argument("Invalid value for --initVal " + std::to_string(arg.initVal));
    }

    if(storeInitData)
    {
        if(arg.initialization == rocblas_initialization_file)
        {
            storeInitData = 0; //Do not store if loading from file
        }
        else
        {
            if(a_file.empty())
                throw std::invalid_argument("Invalid value for --a_file " + a_file);
            if(b_file.empty())
                throw std::invalid_argument("Invalid value for --b_file " + b_file);
            if(c_file.empty())
                throw std::invalid_argument("Invalid value for --c_file " + c_file);
        }
    }

    if(storeOutputData)
    {
        if(o_file.empty())
            throw std::invalid_argument("Invalid value for --o_file " + o_file);
    }

    // Device Query
    rocblas_int device_count = query_device_property();

    if(device_count <= device_id || device_count < multi_device || (multi_device>1 && device_id))
        throw std::invalid_argument("Invalid Device ID");

    if(multi_device>1 && arg.time_each_iter)
        throw std::invalid_argument("Cannot combine multi_device and time_each_iter");

    if(multi_device > 1)
    {
        perfBarrier.init(multi_device);
        memBarrier.init(multi_device-1);
        memBarrier2.init(multi_device-1);
    }
    else
        set_device(device_id);
}

template <typename Ti, typename To = Ti, typename Ui, typename Uo>
void loadFromBin(Ui&               hA,
                 std::string       ADataFile,
                 Ui&               hB,
                 std::string       BDataFile,
                 Uo&               hC,
                 std::string       CDataFile)
{
    {
        size_t sz = hA.lda() * hA.n() * sizeof(Ti) * hA.batch_count();
        std::ifstream FILE(ADataFile, std::ios::in | std::ofstream::binary);
        FILE.seekg(0, FILE.end);
        int fileLength = FILE.tellg();
        FILE.seekg(0, FILE.beg);
        auto* A   = hA[0];


        if(sz > fileLength)
        {
            std::cout << "Binary file bytes " << fileLength << " Gemm required bytes " << sz
                      << std::endl;
            std::cout << "Not enough elements in A data file...exiting" << std::endl;
            exit(1);
        }
        FILE.read(reinterpret_cast<char*>(&A[0]), sz);
    }

    {
        size_t sz = hB.lda() * hB.n() * sizeof(Ti) * hB.batch_count();
        std::ifstream FILE(BDataFile, std::ios::in | std::ofstream::binary);
        FILE.seekg(0, FILE.end);
        int fileLength = FILE.tellg();
        FILE.seekg(0, FILE.beg);
        auto* B   = hB[0];

        if(sz > fileLength)
        {
            std::cout << "Binary file bytes " << fileLength << " Gemm required bytes " << sz
                      << std::endl;
            std::cout << "Not enough elements in B data file...exiting" << std::endl;
            exit(1);
        }
        FILE.read(reinterpret_cast<char*>(&B[0]), sz);
    }

    {
        size_t sz = hC.lda() * hC.n() * sizeof(Ti) * hC.batch_count();
        std::ifstream FILE(CDataFile, std::ios::in | std::ofstream::binary);
        FILE.seekg(0, FILE.end);
        int fileLength = FILE.tellg();
        FILE.seekg(0, FILE.beg);
        auto* C   = hC[0];

        if(sz > fileLength)
        {
            std::cout << "Binary file bytes " << fileLength << " Gemm required bytes " << sz
                      << std::endl;
            std::cout << "Not enough elements in C data file...exiting" << std::endl;
            exit(1);
        }
        FILE.read(reinterpret_cast<char*>(&C[0]), sz);
    }
}

template <typename Ti, typename To>
void storeInitToBin(host_strided_batch_matrix<Ti>&                hA,
                    std::string       ADataFile,
                    host_strided_batch_matrix<Ti>&                hB,
                    std::string       BDataFile,
                    host_strided_batch_matrix<To>&               hC,
                    std::string       CDataFile)
{
    std::string preFix;
    if(multi_device>1)
    {
        int deviceId;
        CHECK_HIP_ERROR(hipGetDevice(&deviceId));
        preFix = "device_" + std::to_string(deviceId) + "_";
    }
    {
        auto* A = hA[0];
        size_t sz = hA.stride() * sizeof(Ti) * hA.batch_count();
        std::ofstream FILE(preFix+ADataFile, std::ios::out | std::ofstream::binary);
        FILE.write(reinterpret_cast<const char*>(&A[0]), sz);
    }

    {
        auto* B = hB[0];
        size_t sz = hB.stride() * sizeof(Ti) * hB.batch_count();
        std::ofstream FILE(preFix+BDataFile, std::ios::out | std::ofstream::binary);
        FILE.write(reinterpret_cast<const char*>(&B[0]), sz);
    }

    {
        auto* C = hC[0];
        size_t        sz = hC.stride() * sizeof(To) * hC.batch_count();
        std::ofstream FILE(preFix+CDataFile, std::ios::out | std::ofstream::binary);
        FILE.write(reinterpret_cast<const char*>(&C[0]), sz);
    }
}

template <typename Ti, typename To>
void storeInitToBin(host_matrix<Ti>&                hA,
                    std::string       ADataFile,
                    host_matrix<Ti>&                hB,
                    std::string       BDataFile,
                    host_matrix<To>&               hC,
                    std::string       CDataFile)
{
    std::string preFix;
    if(multi_device>1)
    {
        int deviceId;
        CHECK_HIP_ERROR(hipGetDevice(&deviceId));
        preFix = "device_" + std::to_string(deviceId) + "_";
    }
    {
        auto* A = hA[0];
        size_t sz = hA.lda() * hA.n() * sizeof(Ti) * hA.batch_count();
        std::ofstream FILE(preFix+ADataFile, std::ios::out | std::ofstream::binary);
        FILE.write(reinterpret_cast<const char*>(&A[0]), sz);
    }

    {
        auto* B = hB[0];
        size_t sz = hB.lda() * hB.n() * sizeof(Ti) * hB.batch_count();
        std::ofstream FILE(preFix+BDataFile, std::ios::out | std::ofstream::binary);
        FILE.write(reinterpret_cast<const char*>(&B[0]), sz);
    }

    {
        auto* C = hC[0];
        size_t        sz = hC.lda() * hC.n() * sizeof(To) * hC.batch_count();
        std::ofstream FILE(preFix+CDataFile, std::ios::out | std::ofstream::binary);
        FILE.write(reinterpret_cast<const char*>(&C[0]), sz);
    }
}

template <typename To>
void storeOutputToBin(host_strided_batch_matrix<To>&   hO,
                      std::string       ODataFile)
{
    std::string preFix;
    if(multi_device>1)
    {
        int deviceId;
        CHECK_HIP_ERROR(hipGetDevice(&deviceId));
        preFix = "device_" + std::to_string(deviceId) + "_";
    }

    {
        auto* O = hO[0];
        size_t        sz = hO.stride() * sizeof(To) * hO.batch_count();
        std::ofstream FILE(preFix+ODataFile, std::ios::out | std::ofstream::binary);
        FILE.write(reinterpret_cast<const char*>(&O[0]), sz);
    }
}

template <typename To>
void storeOutputToBin(host_matrix<To>&   hO,
                      std::string       ODataFile)
{
    std::string preFix;
    if(multi_device>1)
    {
        int deviceId;
        CHECK_HIP_ERROR(hipGetDevice(&deviceId));
        preFix = "device_" + std::to_string(deviceId) + "_";
    }

    {
        auto* O = hO[0];
        size_t        sz = hO.lda() * hO.n() * sizeof(To) * hO.batch_count();
        std::ofstream FILE(preFix+ODataFile, std::ios::out | std::ofstream::binary);
        FILE.write(reinterpret_cast<const char*>(&O[0]), sz);
    }
}

template <typename To>
void storeOutputToBin(rocblas_int       N,
                host_vector<To>&   hO,
                rocblas_int       ldo,
                std::string       ODataFile,
                rocblas_int       batch_count)
{
    std::string preFix;
    if(multi_device>1)
    {
        int deviceId;
        CHECK_HIP_ERROR(hipGetDevice(&deviceId));
        preFix = "device_" + std::to_string(deviceId) + "_";
    }

    {
        size_t        sz = ldo * N * sizeof(To) * batch_count;
        std::ofstream FILE(preFix+ODataFile, std::ios::out | std::ofstream::binary);
        FILE.write(reinterpret_cast<const char*>(&hO[0]), sz);
    }
}

static std::mt19937 rng;

template <typename>
struct FP_PARAM;

template <>
struct FP_PARAM<double>
{
    using UINT_T              = uint64_t;
    static constexpr int NSIG = 52;
    static constexpr int NEXP = 11;
};

template <>
struct FP_PARAM<float>
{
    using UINT_T              = uint32_t;
    static constexpr int NSIG = 23;
    static constexpr int NEXP = 8;
};

template <>
struct FP_PARAM<rocblas_bfloat16>
{
    using UINT_T              = uint16_t;
    static constexpr int NSIG = 7;
    static constexpr int NEXP = 8;
};

template <>
struct FP_PARAM<rocblas_half>
{
    using UINT_T              = uint16_t;
    static constexpr int NSIG = 10;
    static constexpr int NEXP = 5;
};

template <typename T>
struct rocm_random_common : FP_PARAM<T>
{
    using typename FP_PARAM<T>::UINT_T;
    using FP_PARAM<T>::NSIG;
    using FP_PARAM<T>::NEXP;
    using random_fp_int_dist = std::uniform_int_distribution<UINT_T>;

    static_assert(sizeof(UINT_T) == sizeof(T), "Type sizes do not match");
    static constexpr UINT_T expmask = (((UINT_T)1 << NEXP) - 1) << NSIG;
    static constexpr UINT_T expbias = ((UINT_T)1 << (NEXP - 1)) - 1;
    static T                signsig_exp(UINT_T signsig, UINT_T exp)
    {
        union
        {
            UINT_T u;
            T      fp;
        };
        u = signsig & ~expmask | (exp + expbias << NSIG) & expmask;
        return fp;
    }
};

template <typename T, int LOW_EXP, int HIGH_EXP>
struct rocm_random : rocm_random_common<T>
{
    using typename rocm_random_common<T>::random_fp_int_dist;
    __attribute__((flatten)) T operator()()
    {
        int exp = std::uniform_int_distribution<int> {}(rng);
        exp     = exp % (HIGH_EXP - LOW_EXP + 1) + LOW_EXP;
        return this->signsig_exp(random_fp_int_dist {}(rng), exp);
    }
};

// These values should be squareable and not overflow/underflow
template <typename T>
struct rocm_random_squareable;

template <>
struct rocm_random_squareable<double> : rocm_random<double, -189, 200>
{
};

template <>
struct rocm_random_squareable<float> : rocm_random<float, -100, 100>
{
};

template <>
struct rocm_random_squareable<rocblas_bfloat16> : rocm_random<rocblas_bfloat16, -100, 100>
{
};

template <>
struct rocm_random_squareable<rocblas_half> : rocm_random<rocblas_half, -100, 100>
{
};

template <typename T>
using rocm_random_full_range = rocm_random<T,
                                           -((int)((uint64_t)1 << (FP_PARAM<T>::NEXP - 1)) - 1),
                                           (uint32_t)((uint64_t)1 << (FP_PARAM<T>::NEXP - 1)) - 1>;

// These values, when scaled, should be addable without loss
// Basically for these values x, a(1+x) should be different than a.
template <typename T>
using rocm_random_addable = rocm_random<T, 1, FP_PARAM<T>::NSIG>;

template <typename T>
struct rocm_random_narrow_range;

template <>
struct rocm_random_narrow_range<double> : rocm_random<double, -189, 0>
{
};

template <>
struct rocm_random_narrow_range<float> : rocm_random<float, -100, 0>
{
};

template <>
struct rocm_random_narrow_range<rocblas_bfloat16> : rocm_random<rocblas_bfloat16, -100, 0>
{
};

template <>
struct rocm_random_narrow_range<rocblas_half> : rocm_random<rocblas_half, -100, 0>
{
};

// Values from 1-2
template <typename T>
using rocm_random_1_2 = rocm_random<T, 0, 1>;

template <template <typename> class RAND, typename T, typename U, std::enable_if_t<!(std::is_same_v<T, int8_t> || std::is_same_v<T, int32_t>),int> = 0>
static void init_matrix( U& mat )
{
    for(int64_t batch_index = 0; batch_index < mat.batch_count(); ++batch_index)
    {
        auto* A   = mat[batch_index];
        auto  M   = mat.m();
        auto  N   = mat.n();
        auto  lda = mat.lda();
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for(size_t i = 0; i < M; ++i)
            for(size_t j = 0; j < N; ++j)
                A[i + j * lda] = RAND<T> {}();
    }
}

template <template <typename> class RAND, typename T, typename U, std::enable_if_t<(std::is_same_v<T, int8_t> || std::is_same_v<T, int32_t>),int> = 0>
static void init_matrix( U& mat )
{
    rocblas_cout << "Invalid init type for int8_t...exiting" << std::endl;
    exit(1);
}

template <typename T, typename U>
static void init_constant_matrix( U&  hA, T val)
{
    for(int64_t batch_index = 0; batch_index < hA.batch_count(); ++batch_index)
    {
        auto* A   = hA[batch_index];
        auto  M   = hA.m();
        auto  N   = hA.n();
        auto  lda = hA.lda();
        for(size_t i = 0; i < M; ++i)
            for(size_t j = 0; j < N; ++j)
                A[i + j * lda] = val;
    }
}

// Absolute value
template <typename T, typename std::enable_if<!rocblas_is_complex<T>, int>::type = 0>
__device__ __host__ inline T rocblas_abs(T x)
{
    return x < 0 ? -x : x;
}

// For complex, we have defined a __device__ __host__ compatible std::abs
template <typename T, typename std::enable_if<rocblas_is_complex<T>, int>::type = 0>
__device__ __host__ inline auto rocblas_abs(T x)
{
    return std::abs(x);
}

// rocblas_half
__device__ __host__ inline rocblas_half rocblas_abs(rocblas_half x)
{
    union
    {
        rocblas_half x;
        uint16_t     data;
    } t = {x};
    t.data &= 0x7fff;
    return t.x;
}

// rocblas_bfloat16 is handled specially
__device__ __host__ inline rocblas_bfloat16 rocblas_abs(rocblas_bfloat16 x)
{
    x.data &= 0x7fff;
    return x;
}

// Output rocblas_half value
inline std::ostream& operator<<(std::ostream& os, rocblas_half x)
{
    return os << float(x);
}

//TODO review
template <typename T, typename U>
void normalizeInputs(U& hA,
                     U& hB)
{
    // We divide each element of B by the maximum corresponding element of A such that elem(A * B) <
    // 2 ** NSIG

    auto  lda = hA.lda();
    auto  ldb = hB.lda();
    for(int64_t batch_index = 0; batch_index < hA.batch_count(); ++batch_index)
    {
        auto* A   = hA[batch_index];
        auto* B   = hB[batch_index];

        for(size_t i = 0; i < hA.n(); ++i)
        {
            T scal = T(0);
            for(size_t j = 0; j < hA.m(); ++j)
            {
                T val = T(rocblas_abs(A[i * lda + j])); 
                if(val > scal)
                    scal = val;
            }

            if(!scal)
                abort();

            scal = T(1) / scal;
            for(size_t k = 0; k < hB.n(); ++k)
                B[k * ldb + i] *= scal;
        }
    }
}

/*! \brief  generate a random NaN number */
template <typename T>
inline T random_nan_generator()
{
    return T(rocblas_nan_rng{});
}

//!
//! @brief enum to check for NaN initialization of the Input vector/matrix
//!
typedef enum rocblas_check_nan_init_
{
    // Alpha sets NaN
    rocblas_client_alpha_sets_nan,

    // Beta sets NaN
    rocblas_client_beta_sets_nan,

    //  Never set NaN
    rocblas_client_never_set_nan

} rocblas_check_nan_init;

//!
//! @brief Initialize a host_strided_batch_matrix.
//! @param hA The host_strided_batch_matrix.
//! @param arg Specifies the argument class.
//! @param nan_init Initialize matrix with Nan's depending upon the rocblas_check_nan_init enum value.
//! @param matrix_type Initialization of the matrix based upon the rocblas_check_matrix_type enum value.
//! @param seedReset reset the seed if true, do not reset the seed otherwise. Use init_cos if seedReset is true else use init_sin.
//! @param alternating_sign Initialize matrix so adjacent entries have alternating sign.
//!
template <typename T, bool altInit = false>
inline void rocblas_init_matrix(host_strided_batch_matrix<T>& hA,
                                const Arguments&              arg,
                                rocblas_check_nan_init        nan_init,
                                rocblas_check_matrix_type     matrix_type,
                                bool                          seedReset        = false,
                                bool                          alternating_sign = false)
{
    if(seedReset)
        rocblas_seedrand();

    if(nan_init == rocblas_client_alpha_sets_nan && rocblas_isnan(arg.alpha))
    {
        rocblas_init_matrix(matrix_type, arg.uplo, random_nan_generator<T>, hA);
    }
    else if(nan_init == rocblas_client_beta_sets_nan && rocblas_isnan(arg.beta))
    {
        rocblas_init_matrix(matrix_type, arg.uplo, random_nan_generator<T>, hA);
    }
    else if(arg.initialization == rocblas_initialization_random_narrow)
    {
        init_matrix<rocm_random_narrow_range, T>(hA);
    }
    else if(arg.initialization == rocblas_initialization_random_broad)
    {
        if(!altInit)
            init_matrix<rocm_random_squareable, T>(hA);
        else
            init_matrix<rocm_random_addable, T>(hA);
    }
    else if(arg.initialization == rocblas_initialization_random_full)
    {
        init_matrix<rocm_random_full_range, T>(hA);
    }
    else if(arg.initialization == rocblas_initialization_const)
    {
        init_constant_matrix<T>(hA, T(arg.initVal));
    }
    else if(arg.initialization == rocblas_initialization_hpl)
    {
        if(alternating_sign)
            rocblas_init_matrix_alternating_sign(
                matrix_type, arg.uplo, random_hpl_generator<T>, hA);
        else
            rocblas_init_matrix(matrix_type, arg.uplo, random_hpl_generator<T>, hA);
    }
    else if(arg.initialization == rocblas_initialization_random_int)
    {
        if(alternating_sign)
            rocblas_init_matrix_alternating_sign(matrix_type, arg.uplo, random_generator<T>, hA);
        else
            rocblas_init_matrix(matrix_type, arg.uplo, random_generator<T>, hA);
    }
    else if(arg.initialization == rocblas_initialization_trig_float)
    {
        rocblas_init_matrix_trig<T>(matrix_type, arg.uplo, hA, seedReset);
    }
    else
    {
        rocblas_cerr << "unknown initialization type" << std::endl;
        rocblas_abort();
    }
}

//!
//! @brief Initialize a host matrix.
//! @param hA The host matrix.
//! @param arg Specifies the argument class.
//! @param nan_init Initialize matrix with Nan's depending upon the rocblas_check_nan_init enum value.
//! @param matrix_type Initialization of the matrix based upon the rocblas_check_matrix_type enum value.
//! @param alternating_sign Initialize matrix so adjacent entries have alternating sign.
//!
template <typename T, bool altInit = false>
inline void rocblas_init_matrix(host_matrix<T>&           hA,
                                const Arguments&          arg,
                                rocblas_check_nan_init    nan_init,
                                rocblas_check_matrix_type matrix_type,
                                bool                      seedReset        = false,
                                bool                      alternating_sign = false)
{
    if(seedReset)
        rocblas_seedrand();

    if(nan_init == rocblas_client_alpha_sets_nan && rocblas_isnan(arg.alpha))
    {
        rocblas_init_matrix(matrix_type, arg.uplo, random_nan_generator<T>, hA);
    }
    else if(nan_init == rocblas_client_beta_sets_nan && rocblas_isnan(arg.beta))
    {
        rocblas_init_matrix(matrix_type, arg.uplo, random_nan_generator<T>, hA);
    }
    else if(arg.initialization == rocblas_initialization_random_narrow)
    {
        init_matrix<rocm_random_narrow_range, T>(hA);
    }
    else if(arg.initialization == rocblas_initialization_random_broad)
    {
        if(!altInit)
            init_matrix<rocm_random_squareable, T>(hA);
        else
            init_matrix<rocm_random_addable, T>(hA);
    }
    else if(arg.initialization == rocblas_initialization_random_full)
    {
        init_matrix<rocm_random_full_range, T>(hA);
    }
    else if(arg.initialization == rocblas_initialization_const)
    {
        init_constant_matrix<T>(hA, T(arg.initVal));
    }
    else if(arg.initialization == rocblas_initialization_hpl)
    {
        if(alternating_sign)
            rocblas_init_matrix_alternating_sign(
                matrix_type, arg.uplo, random_hpl_generator<T>, hA);
        else
            rocblas_init_matrix(matrix_type, arg.uplo, random_hpl_generator<T>, hA);
    }
    else if(arg.initialization == rocblas_initialization_random_int)
    {
        if(alternating_sign)
            rocblas_init_matrix_alternating_sign(matrix_type, arg.uplo, random_generator<T>, hA);
        else
            rocblas_init_matrix(matrix_type, arg.uplo, random_generator<T>, hA);
    }
    else if(arg.initialization == rocblas_initialization_trig_float)
    {
        rocblas_init_matrix_trig<T>(matrix_type, arg.uplo, hA, seedReset);
    }
    else
    {
        rocblas_cerr << "unknown initialization type" << std::endl;
        rocblas_abort();
    }

}

#endif /* _UTILITY_ */
