#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <stdbool.h>

// SIMD支持
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#define USE_NEON
#elif defined(__AVX__)
#include <immintrin.h>
#define USE_AVX
#elif defined(__SSE3__)
#include <pmmintrin.h>
#define USE_SSE3
#endif

// OpenMP支持
#ifdef _OPENMP
#include <omp.h>
#endif

#define PUSH_CONST 0x01
#define MATMUL 0x02
#define ADD 0x03
#define SIGMOID 0x04
#define ODE_SOLVE 0x05
#define RETURN 0xFF

// 内存对齐设置
#define ALIGN_BYTES 32
// 在ARM上使用标准内存分配函数替代_mm_malloc
#define ALIGNED_MALLOC(size) malloc(size)
#define ALIGNED_FREE(ptr) free(ptr)

// 内存池大小设置
#define MEMORY_POOL_SIZE 32

/* Structure for VM constants */
typedef struct {
    float *data;
    size_t rows;
    size_t cols;
    bool from_pool;  // 标记是否来自内存池
} Matrix;

/* Memory pool structure */
typedef struct {
    Matrix matrices[MEMORY_POOL_SIZE];
    float *data_blocks[MEMORY_POOL_SIZE];
    bool used[MEMORY_POOL_SIZE];
    size_t size;
} MatrixPool;

/* Global memory pool */
static MatrixPool g_pool = {0};

/* 函数前向声明 */
Matrix *matrix_create(size_t rows, size_t cols);
void matrix_free(Matrix *m);
Matrix *matrix_multiply(const Matrix *A, const Matrix *B);
Matrix *matrix_add(const Matrix *A, const Matrix *B);
Matrix *matrix_sigmoid(const Matrix *A);
Matrix *matrix_divide(const Matrix *A, const Matrix *B);
Matrix *matrix_multiply_elementwise(const Matrix *A, const Matrix *B);
Matrix *matrix_concat(const Matrix *A, const Matrix *B);

/* Initialize memory pool */
void pool_init(size_t max_elements) {
    if (g_pool.size > 0) return; // 已初始化
    
    g_pool.size = MEMORY_POOL_SIZE;
    memset(g_pool.used, 0, sizeof(g_pool.used));
    memset(g_pool.data_blocks, 0, sizeof(g_pool.data_blocks));
}

/* Get matrix from pool */
Matrix *pool_get_matrix(size_t rows, size_t cols) {
    size_t data_size = rows * cols * sizeof(float);
    
    // 寻找可用的矩阵槽位
    for (size_t i = 0; i < g_pool.size; i++) {
        if (!g_pool.used[i]) {
            if (g_pool.data_blocks[i] == NULL || 
                rows * cols > g_pool.matrices[i].rows * g_pool.matrices[i].cols) {
                // 需要分配新内存或重新分配更大内存
                if (g_pool.data_blocks[i]) {
                    ALIGNED_FREE(g_pool.data_blocks[i]);
                }
                #ifdef USE_AVX
                g_pool.data_blocks[i] = (float*)ALIGNED_MALLOC(data_size);
                #else
                g_pool.data_blocks[i] = (float*)calloc(rows * cols, sizeof(float));
                #endif
                if (!g_pool.data_blocks[i]) return NULL;
            }
            
            // 初始化矩阵
            Matrix *m = &g_pool.matrices[i];
            m->data = g_pool.data_blocks[i];
            m->rows = rows;
            m->cols = cols;
            m->from_pool = true;
            
            // 标记为使用中
            g_pool.used[i] = true;
            
            // 清零数据
            memset(m->data, 0, data_size);
            return m;
        }
    }
    
    // 池已满，回落到普通分配
    return matrix_create(rows, cols);
}

/* Return matrix to pool */
void pool_return_matrix(Matrix *m) {
    if (!m || !m->from_pool) return;
    
    // 找到矩阵在池中的位置
    for (size_t i = 0; i < g_pool.size; i++) {
        if (&g_pool.matrices[i] == m) {
            g_pool.used[i] = false;
            return;
        }
    }
}

/* VM state */
typedef struct {
    float *hidden_state;
    size_t hidden_size;
} VMState;

/* Allocate a matrix */
Matrix *matrix_create(size_t rows, size_t cols) {
    // 先尝试从内存池获取
    if (g_pool.size > 0) {
        Matrix *m = pool_get_matrix(rows, cols);
        if (m) return m;
    }
    
    // 内存池没有可用的矩阵，使用传统分配
    Matrix *m = (Matrix *)malloc(sizeof(Matrix));
    if (!m) return NULL;
    
    #ifdef USE_AVX
    m->data = (float*)ALIGNED_MALLOC(rows * cols * sizeof(float));
    #else
    m->data = (float *)calloc(rows * cols, sizeof(float));
    #endif
    
    if (!m->data) {
        free(m);
        return NULL;
    }
    
    m->rows = rows;
    m->cols = cols;
    m->from_pool = false;
    return m;
}

/* Free a matrix */
void matrix_free(Matrix *m) {
    if (!m) return;
    
    // 如果矩阵来自内存池，则归还到池中
    if (m->from_pool) {
        pool_return_matrix(m);
        return;
    }
    
    // 否则传统释放
    #ifdef USE_AVX
    ALIGNED_FREE(m->data);
    #else
    free(m->data);
    #endif
    free(m);
}

/* Matrix multiplication: C = A * B */
Matrix *matrix_multiply(const Matrix *A, const Matrix *B) {
    if (A->cols != B->rows) return NULL;
    
    Matrix *C = matrix_create(A->rows, B->cols);
    if (!C) return NULL;
    
    const size_t M = A->rows;
    const size_t N = B->cols;
    const size_t K = A->cols;

#ifdef USE_NEON
    // 使用ARM NEON SIMD指令优化
    const size_t vec_size = 4; // NEON可以同时处理4个单精度浮点数
    
    // 分块乘法提高缓存命中率
    #pragma omp parallel for collapse(2) if(M*N*K > 1000000)
    for (size_t i = 0; i < M; i++) {
        for (size_t j = 0; j < N; j += vec_size) {
            // 处理每行4个元素一组
            if (j + vec_size <= N) {
                // 初始化结果向量为0
                float32x4_t c_vec = vdupq_n_f32(0.0f);
                
                // 逐个乘加累积
                for (size_t k = 0; k < K; k++) {
                    // 加载A的一个元素并广播到4个位置
                    float32x4_t a_val = vdupq_n_f32(A->data[i * K + k]);
                    
                    // 加载B的4个元素 (B在这里按列优先存储)
                    float32x4_t b_vals = vld1q_f32(&B->data[k * N + j]);
                    
                    // 执行乘加操作: c += a * b
                    c_vec = vmlaq_f32(c_vec, a_val, b_vals);
                }
                
                // 存储结果
                vst1q_f32(&C->data[i * N + j], c_vec);
            }
            else {
                // 处理不足4个元素的余数部分
                for (size_t jj = j; jj < N; jj++) {
                    float sum = 0.0f;
                    for (size_t k = 0; k < K; k++) {
                        sum += A->data[i * K + k] * B->data[k * N + jj];
                    }
                    C->data[i * N + jj] = sum;
                }
            }
        }
    }
#elif defined(USE_AVX)
    // 使用AVX指令集优化
    const size_t block_size = 4; // AVX可以同时处理8个单精度浮点数
    
    // 分块乘法提高缓存命中率
    #pragma omp parallel for collapse(2) if(M*N*K > 1000000)
    for (size_t i = 0; i < M; i += block_size) {
        for (size_t j = 0; j < N; j += block_size) {
            for (size_t k = 0; k < K; k++) {
                size_t i_max = (i + block_size < M) ? i + block_size : M;
                size_t j_max = (j + block_size < N) ? j + block_size : N;
                
                for (size_t ii = i; ii < i_max; ii++) {
                    // 广播A的当前元素到向量寄存器
                    __m256 a_val = _mm256_set1_ps(A->data[ii * K + k]);
                    
                    // 对B的行进行向量化处理
                    for (size_t jj = j; jj < j_max; jj += 8) {
                        if (jj + 8 <= j_max) {
                            // 加载B的8个元素
                            __m256 b_vals = _mm256_loadu_ps(&B->data[k * N + jj]);
                            // 加载C的当前8个元素
                            __m256 c_vals = _mm256_loadu_ps(&C->data[ii * N + jj]);
                            // 计算a*b并累加到c
                            c_vals = _mm256_add_ps(c_vals, _mm256_mul_ps(a_val, b_vals));
                            // 存回C
                            _mm256_storeu_ps(&C->data[ii * N + jj], c_vals);
                        } else {
                            // 处理剩余不满8个的元素
                            for (size_t j_remainder = jj; j_remainder < j_max; j_remainder++) {
                                C->data[ii * N + j_remainder] += A->data[ii * K + k] * B->data[k * N + j_remainder];
                            }
                            break;
                        }
                    }
                }
            }
        }
    }
#elif defined(USE_SSE3)
    // 使用SSE3指令集优化
    const size_t block_size = 4; // SSE可以同时处理4个单精度浮点数
    
    #pragma omp parallel for collapse(2) if(M*N*K > 1000000)
    for (size_t i = 0; i < M; i += block_size) {
        for (size_t j = 0; j < N; j += block_size) {
            for (size_t k = 0; k < K; k++) {
                size_t i_max = (i + block_size < M) ? i + block_size : M;
                size_t j_max = (j + block_size < N) ? j + block_size : N;
                
                for (size_t ii = i; ii < i_max; ii++) {
                    // 广播A的当前元素到向量寄存器
                    __m128 a_val = _mm_set1_ps(A->data[ii * K + k]);
                    
                    // 对B的行进行向量化处理
                    for (size_t jj = j; jj < j_max; jj += 4) {
                        if (jj + 4 <= j_max) {
                            // 加载B的4个元素
                            __m128 b_vals = _mm_loadu_ps(&B->data[k * N + jj]);
                            // 加载C的当前4个元素
                            __m128 c_vals = _mm_loadu_ps(&C->data[ii * N + jj]);
                            // 计算a*b并累加到c
                            c_vals = _mm_add_ps(c_vals, _mm_mul_ps(a_val, b_vals));
                            // 存回C
                            _mm_storeu_ps(&C->data[ii * N + jj], c_vals);
                        } else {
                            // 处理剩余不满4个的元素
                            for (size_t j_remainder = jj; j_remainder < j_max; j_remainder++) {
                                C->data[ii * N + j_remainder] += A->data[ii * K + k] * B->data[k * N + j_remainder];
                            }
                            break;
                        }
                    }
                }
            }
        }
    }
#else
    // 基础分块优化，无SIMD
    const size_t block_size = 32; // 缓存友好的分块大小
    
    #pragma omp parallel for collapse(2) if(M*N*K > 1000000)
    for (size_t i = 0; i < M; i += block_size) {
        for (size_t j = 0; j < N; j += block_size) {
            for (size_t k = 0; k < K; k += block_size) {
                size_t i_max = (i + block_size < M) ? i + block_size : M;
                size_t j_max = (j + block_size < N) ? j + block_size : N;
                size_t k_max = (k + block_size < K) ? k + block_size : K;
                
                for (size_t ii = i; ii < i_max; ii++) {
                    for (size_t kk = k; kk < k_max; kk++) {
                        float a_val = A->data[ii * K + kk];
                        for (size_t jj = j; jj < j_max; jj++) {
                            C->data[ii * N + jj] += a_val * B->data[kk * N + jj];
                        }
                    }
                }
            }
        }
    }
#endif

    return C;
}

/* Matrix addition: C = A + B (B is broadcasted if scalar-like) */
Matrix *matrix_add(const Matrix *A, const Matrix *B) {
    Matrix *C = matrix_create(A->rows, A->cols);
    if (!C) return NULL;
    
    const size_t n = A->rows * A->cols;
    const bool is_scalar = (B->rows * B->cols == 1);
    
    if (is_scalar) {
        // B是标量情况的广播加法
        const float b_val = B->data[0];
        
#ifdef USE_NEON
        // NEON优化
        if (n >= 4) {
            const size_t vec_size = 4;
            const size_t vec_count = n / vec_size;
            const size_t remainder = n % vec_size;
            
            // 加载广播值
            const float32x4_t b_vec = vdupq_n_f32(b_val);
            
            #pragma omp parallel for if(n > 10000)
            for (size_t i = 0; i < vec_count; i++) {
                const size_t idx = i * vec_size;
                float32x4_t a_vec = vld1q_f32(&A->data[idx]);
                float32x4_t result = vaddq_f32(a_vec, b_vec);
                vst1q_f32(&C->data[idx], result);
            }
            
            // 处理剩余元素
            for (size_t i = n - remainder; i < n; i++) {
                C->data[i] = A->data[i] + b_val;
            }
            
            return C;
        }
#elif defined(USE_AVX)
        // AVX优化
        if (n >= 8) {
            const size_t vec_size = 8;
            const size_t vec_count = n / vec_size;
            const size_t remainder = n % vec_size;
            
            // 加载广播值
            const __m256 b_vec = _mm256_set1_ps(b_val);
            
            #pragma omp parallel for if(n > 10000)
            for (size_t i = 0; i < vec_count; i++) {
                const size_t idx = i * vec_size;
                __m256 a_vec = _mm256_loadu_ps(&A->data[idx]);
                __m256 result = _mm256_add_ps(a_vec, b_vec);
                _mm256_storeu_ps(&C->data[idx], result);
            }
            
            // 处理剩余元素
            for (size_t i = n - remainder; i < n; i++) {
                C->data[i] = A->data[i] + b_val;
            }
            
            return C;
        }
#elif defined(USE_SSE3)
        // SSE优化
        if (n >= 4) {
            const size_t vec_size = 4;
            const size_t vec_count = n / vec_size;
            const size_t remainder = n % vec_size;
            
            // 加载广播值
            const __m128 b_vec = _mm_set1_ps(b_val);
            
            #pragma omp parallel for if(n > 10000)
            for (size_t i = 0; i < vec_count; i++) {
                const size_t idx = i * vec_size;
                __m128 a_vec = _mm_loadu_ps(&A->data[idx]);
                __m128 result = _mm_add_ps(a_vec, b_vec);
                _mm_storeu_ps(&C->data[idx], result);
            }
            
            // 处理剩余元素
            for (size_t i = n - remainder; i < n; i++) {
                C->data[i] = A->data[i] + b_val;
            }
            
            return C;
        }
#endif
        
        // 标量实现(小矩阵或无SIMD)
        #pragma omp parallel for if(n > 10000)
        for (size_t i = 0; i < n; i++) {
            C->data[i] = A->data[i] + b_val;
        }
    } else {
        // B非标量情况，元素对元素加法
#ifdef USE_NEON
        // NEON优化
        if (n >= 4) {
            const size_t vec_size = 4;
            const size_t vec_count = n / vec_size;
            const size_t remainder = n % vec_size;
            
            #pragma omp parallel for if(n > 10000)
            for (size_t i = 0; i < vec_count; i++) {
                const size_t idx = i * vec_size;
                float32x4_t a_vec = vld1q_f32(&A->data[idx]);
                float32x4_t b_vec = vld1q_f32(&B->data[idx]);
                float32x4_t result = vaddq_f32(a_vec, b_vec);
                vst1q_f32(&C->data[idx], result);
            }
            
            // 处理剩余元素
            for (size_t i = n - remainder; i < n; i++) {
                C->data[i] = A->data[i] + B->data[i];
            }
            
            return C;
        }
#elif defined(USE_AVX)
        // AVX优化
        if (n >= 8) {
            const size_t vec_size = 8;
            const size_t vec_count = n / vec_size;
            const size_t remainder = n % vec_size;
            
            #pragma omp parallel for if(n > 10000)
            for (size_t i = 0; i < vec_count; i++) {
                const size_t idx = i * vec_size;
                __m256 a_vec = _mm256_loadu_ps(&A->data[idx]);
                __m256 b_vec = _mm256_loadu_ps(&B->data[idx]);
                __m256 result = _mm256_add_ps(a_vec, b_vec);
                _mm256_storeu_ps(&C->data[idx], result);
            }
            
            // 处理剩余元素
            for (size_t i = n - remainder; i < n; i++) {
                C->data[i] = A->data[i] + B->data[i];
            }
            
            return C;
        }
#elif defined(USE_SSE3)
        // SSE优化
        if (n >= 4) {
            const size_t vec_size = 4;
            const size_t vec_count = n / vec_size;
            const size_t remainder = n % vec_size;
            
            #pragma omp parallel for if(n > 10000)
            for (size_t i = 0; i < vec_count; i++) {
                const size_t idx = i * vec_size;
                __m128 a_vec = _mm_loadu_ps(&A->data[idx]);
                __m128 b_vec = _mm_loadu_ps(&B->data[idx]);
                __m128 result = _mm_add_ps(a_vec, b_vec);
                _mm_storeu_ps(&C->data[idx], result);
            }
            
            // 处理剩余元素
            for (size_t i = n - remainder; i < n; i++) {
                C->data[i] = A->data[i] + B->data[i];
            }
            
            return C;
        }
#endif
        
        // 标量实现
        #pragma omp parallel for if(n > 10000)
        for (size_t i = 0; i < n; i++) {
            C->data[i] = A->data[i] + B->data[i];
        }
    }
    
    return C;
}

#ifdef USE_SSE3
/* Fast vectorized exponential approximation for SSE */
static inline __m128 exp_ps(__m128 x) {
    // 近似计算exp(x)，适用于SSE
    // 通过多项式近似实现
    __m128 one = _mm_set1_ps(1.0f);
    __m128 half = _mm_set1_ps(0.5f);
    
    // 限制输入范围以提高精度
    x = _mm_min_ps(_mm_max_ps(x, _mm_set1_ps(-88.0f)), _mm_set1_ps(88.0f));
    
    // 近似计算exp(x)
    // 使用泰勒级数展开: 1 + x + x^2/2 + x^3/6 + ...
    __m128 result = one;
    __m128 term = one;
    __m128 factorial = one;
    
    // 计算前5项
    for (int i = 1; i <= 5; i++) {
        term = _mm_mul_ps(term, _mm_div_ps(x, _mm_set1_ps((float)i)));
        result = _mm_add_ps(result, term);
    }
    
    return result;
}
#endif

/* Accurate sigmoid implementation */
static float accurate_sigmoid(float x) {
    // 限制输入范围避免溢出
    x = fminf(fmaxf(x, -88.0f), 88.0f);
    // 使用标准sigmoid公式: 1/(1+exp(-x))
    return 1.0f / (1.0f + expf(-x));
}

/* Fast sigmoid approximation - 保留但不使用 */
static float fast_sigmoid(float x) {
    // 快速近似计算sigmoid: 1/(1+exp(-x))
    x = fminf(fmaxf(x, -10.0f), 10.0f); // 限制范围避免精度损失
    // 使用分段近似计算
    if (x > 5.0f) return 0.99f;
    if (x < -5.0f) return 0.01f;
    return 0.5f + 0.25f * x * (1.0f - 0.05f * x * x);
}

/* NEON向量化exp函数实现 */
static inline float32x4_t exp_ps_neon(float32x4_t x) {
    // 限制范围
    const float32x4_t max_logf = vdupq_n_f32(88.3762626647949f);
    const float32x4_t min_logf = vdupq_n_f32(-88.3762626647949f);
    
    x = vminq_f32(vmaxq_f32(x, min_logf), max_logf);
    
    // 使用自然对数的e为底数转换: e^x = 2^(x * log2(e))
    const float32x4_t log2e = vdupq_n_f32(1.44269504088896341f);
    x = vmulq_f32(x, log2e);
    
    // 分离整数和小数部分 - 使用简单方法实现，避免非标准函数
    // 直接转换为整数再转回浮点，实现floor
    int32x4_t x_int = vcvtq_s32_f32(x);
    float32x4_t fx = vcvtq_f32_s32(x_int);
    
    // 确保对负数正确处理：如果x < fx，则fx = fx - 1
    uint32x4_t mask = vcltq_f32(x, fx);
    int32x4_t one_int = vdupq_n_s32(1);
    int32x4_t fx_adj = vsubq_s32(vcvtq_s32_f32(fx), vandq_s32(vreinterpretq_s32_u32(mask), one_int));
    fx = vcvtq_f32_s32(fx_adj);
    
    float32x4_t cx = vsubq_f32(x, fx);
    
    // 计算2^小数部分，使用多项式近似
    const float32x4_t c1 = vdupq_n_f32(0.693359375f);
    const float32x4_t c2 = vdupq_n_f32(-2.12194440e-4f);
    float32x4_t y = vmulq_f32(cx, c1);
    y = vmlaq_f32(y, cx, c2);
    
    // 使用多项式近似计算exp(y)
    const float32x4_t one = vdupq_n_f32(1.0f);
    const float32x4_t coef1 = vdupq_n_f32(0.5f);
    const float32x4_t coef2 = vdupq_n_f32(0.1666666666f);
    const float32x4_t coef3 = vdupq_n_f32(0.0416666666f);
    
    // 泰勒级数展开: 1 + y + y^2/2! + y^3/3! + ...
    float32x4_t result = one;
    float32x4_t yp = y;
    result = vaddq_f32(result, yp);
    
    yp = vmulq_f32(yp, y);
    result = vmlaq_f32(result, yp, coef1);
    
    yp = vmulq_f32(yp, y);
    result = vmlaq_f32(result, yp, coef2);
    
    yp = vmulq_f32(yp, y);
    result = vmlaq_f32(result, yp, coef3);
    
    // 使用指数部分位移
    int32x4_t emm0 = vcvtq_s32_f32(fx);
    emm0 = vaddq_s32(emm0, vdupq_n_s32(0x7f));
    emm0 = vshlq_n_s32(emm0, 23);
    float32x4_t pow2n = vreinterpretq_f32_s32(emm0);
    
    // 最终结果: result * 2^n
    result = vmulq_f32(result, pow2n);
    
    return result;
}

/* Sigmoid activation with optimization */
Matrix *matrix_sigmoid(const Matrix *A) {
    Matrix *C = matrix_create(A->rows, A->cols);
    if (!C) return NULL;
    
    const size_t n = A->rows * A->cols;

#ifdef USE_NEON
    // NEON优化实现
    if (n >= 4) {
        const size_t vec_size = 4; // NEON处理4个float
        const size_t vec_count = n / vec_size;
        const size_t remainder = n % vec_size;
        
        // 预计算常量向量
        const float32x4_t ones = vdupq_n_f32(1.0f);
        const float32x4_t neg_limit = vdupq_n_f32(-88.0f);
        const float32x4_t pos_limit = vdupq_n_f32(88.0f);
        
        // 向量化处理主要部分
        #pragma omp parallel for if(n > 10000)
        for (size_t i = 0; i < vec_count; i++) {
            const size_t idx = i * vec_size;
            
            // 加载4个元素
            float32x4_t x = vld1q_f32(&A->data[idx]);
            
            // 限制范围
            x = vminq_f32(vmaxq_f32(x, neg_limit), pos_limit);
            
            // 计算 -x
            float32x4_t neg_x = vnegq_f32(x);
            
            // exp(-x)
            float32x4_t exp_neg_x = exp_ps_neon(neg_x);
            
            // 1 + exp(-x)
            float32x4_t denominator = vaddq_f32(ones, exp_neg_x);
            
            // 1.0 / (1.0 + exp(-x))
            float32x4_t result = vdivq_f32(ones, denominator);
            
            // 存储结果
            vst1q_f32(&C->data[idx], result);
        }
        
        // 处理剩余元素
        for (size_t i = n - remainder; i < n; i++) {
            C->data[i] = accurate_sigmoid(A->data[i]);
        }
        
        return C;
    }
#elif defined(USE_AVX)
    // AVX optimization for large matrices
    if (n >= 16) {
        const size_t vec_size = 8;
        const size_t vec_count = n / vec_size;
        const size_t remainder = n % vec_size;
        
        // 处理剩余元素
        for (size_t i = n - remainder; i < n; i++) {
            C->data[i] = accurate_sigmoid(A->data[i]);
        }
        
        return C;
    }
#elif defined(USE_SSE3)
    // SSE optimization for larger matrices
    if (n >= 8) {
        const size_t vec_size = 4;
        const size_t vec_count = n / vec_size;
        const size_t remainder = n % vec_size;
        
        // 处理剩余元素
        for (size_t i = n - remainder; i < n; i++) {
            C->data[i] = accurate_sigmoid(A->data[i]);
        }
        
        return C;
    }
#endif

    // Scalar implementation for small matrices or fallback
    #pragma omp parallel for if(n > 10000)
    for (size_t i = 0; i < n; i++) {
        C->data[i] = accurate_sigmoid(A->data[i]);
    }
    
    return C;
}

/* Element-wise division: C = A / B (B is broadcasted) */
Matrix *matrix_divide(const Matrix *A, const Matrix *B) {
    Matrix *C = matrix_create(A->rows, A->cols);
    for (size_t i = 0; i < A->rows * A->cols; i++) {
        float b_val = (B->rows * B->cols == 1) ? B->data[0] : B->data[i];
        if (b_val == 0.0f) return NULL;
        C->data[i] = A->data[i] / b_val;
    }
    return C;
}

/* Element-wise multiplication: C = A * B */
Matrix *matrix_multiply_elementwise(const Matrix *A, const Matrix *B) {
    Matrix *C = matrix_create(A->rows, A->cols);
    for (size_t i = 0; i < A->rows * A->cols; i++) {
        float b_val = (B->rows * B->cols == 1) ? B->data[0] : B->data[i];
        C->data[i] = A->data[i] * b_val;
    }
    return C;
}

/* Concatenate matrices A and B horizontally */
Matrix *matrix_concat(const Matrix *A, const Matrix *B) {
    if (A->rows != B->rows) return NULL;
    Matrix *C = matrix_create(A->rows, A->cols + B->cols);
    for (size_t i = 0; i < A->rows; i++) {
        memcpy(C->data + i * C->cols, A->data + i * A->cols, A->cols * sizeof(float));
        memcpy(C->data + i * C->cols + A->cols, B->data + i * B->cols, B->cols * sizeof(float));
    }
    return C;
}

/* LTC ODE solver: dh/dt = (-h + (W @ combined + b) * gate) / tau */
Matrix *ltc_ode_solver(const Matrix *h, const Matrix *x, const Matrix *consts[], size_t const_indices[]) {
    size_t w_gate_idx = const_indices[0];
    size_t b_gate_idx = const_indices[1];
    size_t w_idx = const_indices[2];
    size_t b_idx = const_indices[3];
    size_t tau_idx = const_indices[4];

    /* combined = [x, h] */
    Matrix *combined = matrix_concat(x, h);
    if (!combined) return NULL;

    /* gate = sigmoid(W_gate @ combined + b_gate) */
    Matrix *gate_intermediate = matrix_multiply(combined, consts[w_gate_idx]);
    Matrix *gate = gate_intermediate ? matrix_add(gate_intermediate, consts[b_gate_idx]) : NULL;
    Matrix *gate_sig = gate ? matrix_sigmoid(gate) : NULL;
    matrix_free(gate_intermediate);
    matrix_free(gate);
    matrix_free(combined);
    if (!gate_sig) return NULL;

    /* W @ combined + b */
    combined = matrix_concat(x, h);
    Matrix *w_out = matrix_multiply(combined, consts[w_idx]);
    Matrix *w_out_b = w_out ? matrix_add(w_out, consts[b_idx]) : NULL;
    matrix_free(w_out);
    matrix_free(combined);
    if (!w_out_b) {
        matrix_free(gate_sig);
        return NULL;
    }

    /* (W @ combined + b) * gate */
    Matrix *gated = matrix_multiply_elementwise(w_out_b, gate_sig);
    matrix_free(w_out_b);
    matrix_free(gate_sig);
    if (!gated) return NULL;

    /* -h + gated */
    Matrix *h_neg = matrix_create(h->rows, h->cols);
    for (size_t i = 0; i < h->rows * h->cols; i++) {
        h_neg->data[i] = -h->data[i];
    }
    Matrix *dhdt = matrix_add(h_neg, gated);
    matrix_free(h_neg);
    matrix_free(gated);
    if (!dhdt) return NULL;

    /* (-h + gated) / tau */
    Matrix *result = matrix_divide(dhdt, consts[tau_idx]);
    matrix_free(dhdt);
    return result;
}

/* Euler ODE solver */
Matrix *euler_solve(Matrix *(*ode_func)(const Matrix *, const Matrix *, const Matrix *[], size_t[]),
                    const Matrix *h0, const Matrix *x, const float *t_eval, size_t t_eval_len,
                    const Matrix *consts[], size_t const_indices[]) {
    // 创建输出矩阵，使用内存池减少内存分配
    Matrix *h = matrix_create(h0->rows, h0->cols);
    if (!h) return NULL;
    
    // 复制初始状态
    memcpy(h->data, h0->data, h0->rows * h0->cols * sizeof(float));
    
    // 预分配导数矩阵，避免在循环中重复分配
    Matrix *dh = NULL;
    size_t h_size = h0->rows * h0->cols;
    
    // 主时间步进循环
    for (size_t i = 0; i < t_eval_len - 1; i++) {
        float dt = t_eval[i + 1] - t_eval[i];
        
        // 计算导数
        dh = ode_func(h, x, consts, const_indices);
        if (!dh) {
            matrix_free(h);
            return NULL;
        }
        
        // 欧拉更新，使用向量化操作
        #ifdef USE_AVX
        if (h_size >= 8) {
            const size_t vec_size = 8;
            const size_t vec_count = h_size / vec_size;
            const size_t remainder = h_size % vec_size;
            const __m256 dt_vec = _mm256_set1_ps(dt);
            
            #pragma omp parallel for if(h_size > 1000)
            for (size_t j = 0; j < vec_count; j++) {
                const size_t idx = j * vec_size;
                __m256 h_vec = _mm256_loadu_ps(&h->data[idx]);
                __m256 dh_vec = _mm256_loadu_ps(&dh->data[idx]);
                __m256 dh_dt = _mm256_mul_ps(dh_vec, dt_vec);
                __m256 result = _mm256_add_ps(h_vec, dh_dt);
                _mm256_storeu_ps(&h->data[idx], result);
            }
            
            // 处理剩余元素
            for (size_t j = h_size - remainder; j < h_size; j++) {
                h->data[j] += dh->data[j] * dt;
            }
        } else 
        #elif defined(USE_SSE3)
        if (h_size >= 4) {
            const size_t vec_size = 4;
            const size_t vec_count = h_size / vec_size;
            const size_t remainder = h_size % vec_size;
            const __m128 dt_vec = _mm_set1_ps(dt);
            
            #pragma omp parallel for if(h_size > 1000)
            for (size_t j = 0; j < vec_count; j++) {
                const size_t idx = j * vec_size;
                __m128 h_vec = _mm_loadu_ps(&h->data[idx]);
                __m128 dh_vec = _mm_loadu_ps(&dh->data[idx]);
                __m128 dh_dt = _mm_mul_ps(dh_vec, dt_vec);
                __m128 result = _mm_add_ps(h_vec, dh_dt);
                _mm_storeu_ps(&h->data[idx], result);
            }
            
            // 处理剩余元素
            for (size_t j = h_size - remainder; j < h_size; j++) {
                h->data[j] += dh->data[j] * dt;
            }
        } else 
        #endif
        {
            // 标量实现
            #pragma omp parallel for if(h_size > 1000)
            for (size_t j = 0; j < h_size; j++) {
                h->data[j] += dh->data[j] * dt;
            }
        }
        
        // 释放导数矩阵
        matrix_free(dh);
    }
    
    return h;
}

/* Check for NaN or Inf */
int is_invalid_float(float x) {
    return isnan(x) || isinf(x);
}

/* Load bytecode from file */
unsigned char *load_bytecode(const char *filename, size_t *length) {
    FILE *f = fopen(filename, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    *length = ftell(f);
    fseek(f, 0, SEEK_SET);
    unsigned char *bytecode = (unsigned char *)malloc(*length);
    fread(bytecode, 1, *length, f);
    fclose(f);
    return bytecode;
}

/* Main VM execution function */
float vm_run(float x_input, float x_mean, float x_std, float y_mean, float y_std,
             float *const_data[], size_t const_rows[], size_t const_cols[], size_t const_count,
             const char *bytecode_file, const float *t_eval, size_t t_eval_len) {
    /* 初始化内存池 */
    pool_init(MEMORY_POOL_SIZE);
             
    /* Normalize input */
    float x_norm = (x_input - x_mean) / x_std;

    /* Create constants */
    Matrix **consts = (Matrix **)malloc(const_count * sizeof(Matrix *));
    for (size_t i = 0; i < const_count; i++) {
        consts[i] = matrix_create(const_rows[i], const_cols[i]);
        memcpy(consts[i]->data, const_data[i], const_rows[i] * const_cols[i] * sizeof(float));
    }

    /* Initialize stack with input: [[x_norm]] */
    Matrix *stack[16]; /* Fixed-size stack for simplicity */
    size_t stack_top = 0;
    stack[stack_top] = matrix_create(1, 1);
    stack[stack_top]->data[0] = x_norm;
    stack_top++;

    /* Load bytecode */
    size_t bytecode_len;
    unsigned char *bytecode = load_bytecode(bytecode_file, &bytecode_len);
    if (!bytecode) {
        for (size_t i = 0; i < stack_top; i++) matrix_free(stack[i]);
        for (size_t i = 0; i < const_count; i++) matrix_free(consts[i]);
        free(consts);
        return NAN;
    }

    size_t ip = 0;
    while (ip < bytecode_len) {
        unsigned char op = bytecode[ip++];

        if (op == PUSH_CONST) {
            size_t idx = bytecode[ip++];
            if (idx >= const_count || stack_top >= 16) goto error;
            stack[stack_top++] = matrix_create(consts[idx]->rows, consts[idx]->cols);
            memcpy(stack[stack_top-1]->data, consts[idx]->data, consts[idx]->rows * consts[idx]->cols * sizeof(float));
        }
        else if (op == MATMUL) {
            if (stack_top < 2) goto error;
            Matrix *w = stack[--stack_top];
            Matrix *x = stack[--stack_top];
            Matrix *result = matrix_multiply(x, w);
            matrix_free(w);
            matrix_free(x);
            if (!result) goto error;
            stack[stack_top++] = result;
        }
        else if (op == ADD) {
            if (stack_top < 2) goto error;
            Matrix *b = stack[--stack_top];
            Matrix *x = stack[--stack_top];
            Matrix *result = matrix_add(x, b);
            matrix_free(b);
            matrix_free(x);
            if (!result) goto error;
            stack[stack_top++] = result;
        }
        else if (op == SIGMOID) {
            if (stack_top < 1) goto error;
            Matrix *x = stack[--stack_top];
            Matrix *result = matrix_sigmoid(x);
            matrix_free(x);
            if (!result) goto error;
            stack[stack_top++] = result;
        }
        else if (op == ODE_SOLVE) {
            if (stack_top < 1 || ip + 7 > bytecode_len) goto error;
            size_t const_indices[5];
            for (size_t i = 0; i < 5; i++) {
                const_indices[i] = bytecode[ip + i];
                if (const_indices[i] >= const_count) goto error;
            }
            size_t w_out_idx = bytecode[ip + 5];
            size_t b_out_idx = bytecode[ip + 6];
            if (w_out_idx >= const_count || b_out_idx >= const_count) goto error;
            ip += 7;

            Matrix *x = stack[stack_top - 1];
            Matrix *h0 = matrix_create(1, consts[const_indices[0]]->cols);
            Matrix *h_final = euler_solve(ltc_ode_solver, h0, x, t_eval, t_eval_len, (const Matrix **)consts, const_indices);
            matrix_free(h0);
            if (!h_final) goto error;

            Matrix *out = matrix_multiply(h_final, consts[w_out_idx]);
            Matrix *out_final = out ? matrix_add(out, consts[b_out_idx]) : NULL;
            matrix_free(out);
            matrix_free(h_final);
            if (!out_final) goto error;

            stack[stack_top++] = out_final;
        }
        else if (op == RETURN) {
            if (stack_top < 1) goto error;
            Matrix *result = stack[--stack_top];
            float output = result->data[0];
            matrix_free(result);

            /* Clean up */
            for (size_t i = 0; i < stack_top; i++) matrix_free(stack[i]);
            for (size_t i = 0; i < const_count; i++) matrix_free(consts[i]);
            free(consts);
            free(bytecode);

            if (is_invalid_float(output)) return NAN;
            return output * y_std + y_mean;
        }
        else {
            goto error;
        }
    }

error:
    for (size_t i = 0; i < stack_top; i++) matrix_free(stack[i]);
    for (size_t i = 0; i < const_count; i++) matrix_free(consts[i]);
    free(consts);
    free(bytecode);
    return NAN;
}