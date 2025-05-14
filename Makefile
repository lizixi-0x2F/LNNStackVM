CC = gcc
CFLAGS = -std=c99 -O3 -Wall -Wextra 
# M系列芯片自带NEON，不需要额外标记
SIMD_FLAGS = 
# Mac环境下暂时移除OpenMP支持
# PARALLEL_FLAGS = -fopenmp
PARALLEL_FLAGS =
LIBS = -lm

# 源代码路径
SRC = src/vm.c
OBJ = $(SRC:.c=.o)
TARGET = libvm.so

# 默认目标
all: $(TARGET)

# 编译动态库
$(TARGET): $(OBJ)
	$(CC) -shared -o $@ $^ $(CFLAGS) $(SIMD_FLAGS) $(PARALLEL_FLAGS) $(LIBS)

# 编译规则
%.o: %.c
	$(CC) -c -fPIC $< -o $@ $(CFLAGS) $(SIMD_FLAGS) $(PARALLEL_FLAGS)

# 清理
clean:
	rm -f $(OBJ) $(TARGET)

.PHONY: all clean
