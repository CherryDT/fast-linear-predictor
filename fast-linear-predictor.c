// fast-linear-predictor.c
// ------------------------------------------------
// Copyright (c) 2025 David Trapp (dt@david-trapp.com)
//
// Licensed under the MIT License. See LICENSE for details.
//
// Crack and predict the next few masked outputs
// of a GF(2)-linear PRNG using per-bit
// Berlekamp–Massey and LFSR stepping with OpenMP.
//
// Usage:  fast-linear-predictor -c count [-b bits] [input_file]
//   -b bits   Number of low-order bits to use (1..64, default 64)
//   -c count  How many future values to predict
//   input_file: one integer per line; stdin otherwise.
//
// Runtime: O(bits * n^2) bit-operations; buffers reused to minimize
// heap overhead and parallelized per-bit with OpenMP.
// ------------------------------------------------

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <omp.h>

#define MAX_BITS 64
#define CHUNK 50000

typedef uint8_t bit;

//------------------------------------------------------------------------------
// Run Berlekamp–Massey on a binary sequence `bits` of length `n`.
// Computes the shortest LFSR (degree L) whose output matches `bits`.
//   - `C` is length-n buffer to receive connection polys (C[0..L])
//   - `B` is length-n buffer for the previous best polynomial
//   - `T` is a temp buffer of length n for copying
// Returns L (the LFSR degree). After, C[0]=1, C[1..L] are taps.
//------------------------------------------------------------------------------
int berlekamp_massey(const bit *bits, int n, bit *C, bit *B, bit *T) {
  // initialize C(x)=1, B(x)=1
  memset(C, 0, n * sizeof(bit));
  memset(B, 0, n * sizeof(bit));
  C[0] = B[0] = 1;

  int L = 0;    // current LFSR degree
  int m = -1;   // last update position

  // main iteration: process bit at position N
  for (int N = 0; N < n; N++) {
    // compute discrepancy d = bits[N] + sum_{i=1..L} C[i]*bits[N-i]
    bit d = bits[N];
    for (int i = 1; i <= L; i++) {
      d ^= C[i] & bits[N - i];
    }

    // if discrepancy non-zero, update polynomials
    if (d) {
      // T(x) = C(x)
      memcpy(T, C, n * sizeof(bit));

      // C(x) = C(x) + x^{N-m} * B(x)
      int shiftN = N - m;
      for (int j = 0; j + shiftN < n; j++) {
        C[j + shiftN] ^= B[j];
      }

      // if 2L <= N, update B←T and degree L
      if (2 * L <= N) {
        memcpy(B, T, n * sizeof(bit));
        L = N + 1 - L;
        m = N;
      }
    }
  }
  return L;
}

//------------------------------------------------------------------------------
// Using the LFSR defined by connection poly C[0..L], generate k future bits.
// `init` holds the last L observed bits (the seed state). `state` is a
// buffer of size >= L+k used to store and step the LFSR. Writes results
// into out[0..k-1].
//------------------------------------------------------------------------------
void predict_bits_reuse(const bit *C, int L, const bit *init,
                        int k, bit *out, bit *state) {
  // load seed state
  memcpy(state, init, L * sizeof(bit));

  // step LFSR k times
  for (int t = 0; t < k; t++) {
    bit fb = 0;
    // compute feedback bit as sum of taps applied to previous state bits
    // new bit = sum_{i=1..L} C[i]*state[position L+t-i]
    for (int i = 1; i <= L; i++) {
      fb ^= C[i] & state[L + t - i];
    }
    // append bit to state and record in out
    state[L + t] = fb;
    out[t] = fb;
  }
}

void usage(char *argv0) {
  fprintf(stderr, "fast-linear-predictor by David Trapp\n");
  fprintf(stderr, "Predicts future outputs of a linear PRNG\n");
  fprintf(stderr, "\n");
  fprintf(stderr, "Usage: %s -c count [-b bits] [input_file]\n", argv0);
  fprintf(stderr, "  -c count  How many future values to predict\n");
  fprintf(stderr, "  -b bits   Number of low-order bits to use (1..%d, default %d)\n", MAX_BITS, MAX_BITS);
  fprintf(stderr, "  input_file: one integer per line; stdin otherwise.\n");
  fprintf(stderr, "\n");
  fprintf(stderr, "The input file should contain one integer (decimal) per line.\n");
  fprintf(stderr, "Outputs the predicted values, one per line.\n");
}

int main(int argc, char *argv[]) {
  int bits = MAX_BITS;
  int predict_count = 0;
  const char *input_file = NULL;

  // parse args
  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "-b") && i+1 < argc) {
      bits = atoi(argv[++i]);
    } else if (!strcmp(argv[i], "-c") && i+1 < argc) {
      predict_count = atoi(argv[++i]);
    } else if (argv[i][0] == '-') {
      usage(argv[0]);
      return 1;
    } else {
      input_file = argv[i];
    }
  }
  if (predict_count < 1) {
    usage(argv[0]);
    return 1;
  }
  if (bits < 1 || bits > MAX_BITS) {
    fprintf(stderr, "Error: bits must be 1..%d\n", MAX_BITS);
    return 1;
  }

  // open file or stdin
  FILE *f = input_file ? fopen(input_file, "r") : stdin;
  if (!f) { perror("fopen"); return 1; }

  // read samples in chunks
  uint64_t *obs = NULL;
  int n = 0, cap = 0;
  while (1) {
    unsigned long long v;
    if (fscanf(f, "%llu", &v) != 1) break;
    if (n == cap) {
      cap += CHUNK;
      uint64_t *tmp = realloc(obs, cap * sizeof(*obs));
      if (!tmp) { perror("realloc"); free(obs); return 1; }
      obs = tmp;
    }
    obs[n++] = v;
  }
  if (f != stdin) fclose(f);
  if (n < 2 * bits) {
    fprintf(stderr, "Need at least %d samples, got %d\n", 2*bits, n);
    free(obs);
    return 1;
  }

  // prepare bit positions [0..bits-1]
  int positions[MAX_BITS];
  for (int b = 0; b < bits; b++) positions[b] = b;

  bit *bits_stream = malloc(n * sizeof(bit));
  bit *C_arr[MAX_BITS], *init_arr[MAX_BITS];
  int L_arr[MAX_BITS];

  // recover each bit-stream in parallel
  #pragma omp parallel
  {
    bit *scratchC = malloc(n * sizeof(bit));
    bit *scratchB = malloc(n * sizeof(bit));
    bit *scratchT = malloc(n * sizeof(bit));
    bit *bs = malloc(n * sizeof(bit));
    #pragma omp for schedule(static)
    for (int i = 0; i < bits; i++) {
      int b = positions[i];
      for (int j = 0; j < n; j++) bs[j] = (obs[j] >> b) & 1;
      memset(scratchC, 0, n * sizeof(bit));
      memset(scratchB, 0, n * sizeof(bit));
      int Li = berlekamp_massey(bs, n, scratchC, scratchB, scratchT);
      L_arr[i] = Li;
      C_arr[i] = malloc((Li + 1) * sizeof(bit));
      memcpy(C_arr[i], scratchC, (Li + 1) * sizeof(bit));
      init_arr[i] = malloc(Li * sizeof(bit));
      memcpy(init_arr[i], bs + (n - Li), Li * sizeof(bit));
    }
    free(bs); free(scratchC); free(scratchB); free(scratchT);
  }
  free(obs);

  // find max L to allocate shared state buffer
  int maxL = 0;
  for (int i = 0; i < bits; i++) if (L_arr[i] > maxL) maxL = L_arr[i];

  bit *out_bits = malloc(predict_count * bits * sizeof(bit));

  // step each LFSR and fill out_bits
  #pragma omp parallel
  {
    bit *state_buf = malloc((maxL + predict_count) * sizeof(bit));
    #pragma omp for schedule(static)
    for (int i = 0; i < bits; i++) {
      predict_bits_reuse(
        C_arr[i], L_arr[i], init_arr[i],
        predict_count,
        out_bits + i * predict_count,
        state_buf
      );
      free(C_arr[i]);
      free(init_arr[i]);
    }
    free(state_buf);
  }

  // reassemble and print predicted values
  for (int k = 0; k < predict_count; k++) {
    uint64_t v = 0;
    for (int i = 0; i < bits; i++) {
      if (out_bits[i * predict_count + k]) {
        v |= (1ULL << positions[i]);
      }
    }
    printf("%llu\n", (unsigned long long)v);
  }

  free(bits_stream);
  free(out_bits);
  return 0;
}
