#ifndef VM_JIT_H
#define VM_JIT_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdint.h>

/* 架构定义 */
#if defined(__x86_64__) || defined(_M_X64)
#define ARCH_X86_64
#elif defined(__aarch64__) || defined(_M_ARM64)
#define ARCH_ARM64
#else
#define ARCH_UNKNOWN
#endif

/* 导出符号定义 */
#ifdef _WIN32
#ifdef VM_JIT_EXPORTS
#define VM_JIT_API __declspec(dllexport)
#else
#define VM_JIT_API __declspec(dllimport)
#endif
#else
#define VM_JIT_API
#endif

/* JIT编译后的代码结构 */
typedef struct {
    void *code;         // 可执行内存
    size_t code_size;   // 代码大小
    void *data;         // 数据部分
    size_t data_size;   // 数据大小
} JITFunction;

/* 导出函数声明 */

/**
 * 使用JIT执行虚拟机
 * @param x_input 输入值
 * @param x_mean 输入均值
 * @param x_std 输入标准差
 * @param y_mean 输出均值
 * @param y_std 输出标准差
 * @param const_data 常量数据
 * @param const_rows 常量行数
 * @param const_cols 常量列数
 * @param const_count 常量数量
 * @param bytecode_file 字节码文件
 * @param t_eval 时间评估点
 * @param t_eval_len 时间评估点数量
 * @return 计算结果
 */
VM_JIT_API float vm_run_jit(float x_input, float x_mean, float x_std, float y_mean, float y_std,
                     float *const_data[], size_t const_rows[], size_t const_cols[], size_t const_count,
                     const char *bytecode_file, const float *t_eval, size_t t_eval_len);

/**
 * 主VM执行函数 - 保持原来API兼容性
 * @param x_input 输入值
 * @param x_mean 输入均值
 * @param x_std 输入标准差
 * @param y_mean 输出均值
 * @param y_std 输出标准差
 * @param const_data 常量数据
 * @param const_rows 常量行数
 * @param const_cols 常量列数
 * @param const_count 常量数量
 * @param bytecode_file 字节码文件
 * @param t_eval 时间评估点
 * @param t_eval_len 时间评估点数量
 * @return 计算结果
 */
VM_JIT_API float vm_run(float x_input, float x_mean, float x_std, float y_mean, float y_std,
                 float *const_data[], size_t const_rows[], size_t const_cols[], size_t const_count,
                 const char *bytecode_file, const float *t_eval, size_t t_eval_len);

/**
 * 使用解释器执行虚拟机
 * @param x_input 输入值
 * @param x_mean 输入均值
 * @param x_std 输入标准差
 * @param y_mean 输出均值
 * @param y_std 输出标准差
 * @param const_data 常量数据
 * @param const_rows 常量行数
 * @param const_cols 常量列数
 * @param const_count 常量数量
 * @param bytecode 字节码
 * @param bytecode_len 字节码长度
 * @param t_eval 时间评估点
 * @param t_eval_len 时间评估点数量
 * @return 计算结果
 */
VM_JIT_API float vm_run_interpreter(float x_input, float x_mean, float x_std, float y_mean, float y_std,
                            float *const_data[], size_t const_rows[], size_t const_cols[], size_t const_count,
                            const unsigned char *bytecode, size_t bytecode_len, const float *t_eval, size_t t_eval_len);

/**
 * 清理JIT分配的资源
 */
VM_JIT_API void vm_cleanup(void);

#ifdef __cplusplus
}
#endif

#endif /* VM_JIT_H */ 