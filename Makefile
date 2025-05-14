CC = gcc
CFLAGS = -O3 -Wall -shared -fPIC -std=c99
ARCH_FLAGS = 
UNAME_S := $(shell uname -s)
UNAME_M := $(shell uname -m)

# 检测架构
ifeq ($(UNAME_M),x86_64)
    ARCH_FLAGS += -DARCH_X86_64 -arch x86_64
else ifeq ($(UNAME_M),arm64)
    ARCH_FLAGS += -DARCH_ARM64 -arch arm64
endif

# 检测操作系统
ifeq ($(UNAME_S),Darwin)
    # macOS特定设置
    LDFLAGS = -lm
    # 使用VM_JIT_EXPORTS宏表示导出符号
    CFLAGS += -DVM_JIT_EXPORTS
else ifeq ($(UNAME_S),Linux)
    # Linux特定设置
    LDFLAGS = -lm
    CFLAGS += -DVM_JIT_EXPORTS
else
    # 默认为Windows
    LDFLAGS = -lm
    CFLAGS += -DVM_JIT_EXPORTS
endif

TARGET_DIR = output
TARGET = $(TARGET_DIR)/lib_vm.so
TARGET_JIT = $(TARGET_DIR)/lib_vm_jit.so

SRCS = src/vm.c
SRCS_JIT = src/vm_jit.c
HEADERS_JIT = src/vm_jit.h

ifeq ($(UNAME_M),x86_64)
    SRCS_JIT += src/vm_jit_x86.c
else ifeq ($(UNAME_M),arm64)
    SRCS_JIT += src/vm_jit_arm64.c
endif

OBJS = $(SRCS:.c=.o)
OBJS_JIT = $(SRCS_JIT:.c=.o)

.PHONY: all clean dirs jit

all: dirs $(TARGET) $(TARGET_JIT)

jit: dirs $(TARGET_JIT)

dirs:
	@mkdir -p $(TARGET_DIR)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) $(ARCH_FLAGS) -o $@ $^ $(LDFLAGS)

$(TARGET_JIT): $(OBJS_JIT)
	$(CC) $(CFLAGS) $(ARCH_FLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) $(ARCH_FLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(OBJS_JIT) $(TARGET) $(TARGET_JIT)