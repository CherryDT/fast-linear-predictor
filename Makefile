# Makefile for fast-linear-predictor

# Compiler settings
CC := gcc
CFLAGS := -O3 -Wall -Wextra -std=c11 -fopenmp

# Target binary
BIN := fast-linear-predictor

# Python test script
PYTHON := python3
PY_TEST := test.py

.PHONY: all test clean

# Default target: build the binary
all: $(BIN)

# Build command
$(BIN): fast-linear-predictor.c
	$(CC) $(CFLAGS) fast-linear-predictor.c -o $(BIN)

# Run tests using the Python script
test: all
	@echo "Running Python tests..."
	$(PYTHON) $(PY_TEST)

# Clean up generated files
clean:
	rm -f $(BIN)
