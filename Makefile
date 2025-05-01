CC = gcc
CFLAGS = -O3 -Wall -shared -fPIC -std=c99 -arch x86_64
TARGET = output/lib_vm.so
SRC = src/vm.c
OBJ = $(SRC:.c=.o)

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CC) $(CFLAGS) -o $@ $^ -lm

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJ) $(TARGET)

.PHONY: all clean