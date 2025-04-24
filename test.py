#!/usr/bin/env python3
import subprocess
import random
import sys
import time

# Path to the binary
BIN = './fast-linear-predictor'
# Number of samples to collect (must be >= 2 * LFSR degree)
SAMPLES = 50000
# Number of values to predict
PREDICT = 16

# Define test scenarios: (label, bits, transform function)
SCENARIOS = [
  ('8-bit',    8, lambda r: r.getrandbits(8)),
  ('low8',     8, lambda r: r.getrandbits(32) & 0xFF),
  ('16-bit',  16, lambda r: r.getrandbits(16)),
  ('32-bit',  32, lambda r: r.getrandbits(32)),
  ('64-bit',  64, lambda r: r.getrandbits(64))
]


def run_scenario(label, bits, transform):
  print(f'\n=== Scenario: {label} (bits={bits}) ===')
  # Initialize RNG with a known seed for reproducibility
  r = random.Random(0xDEADBEEF)

  # Collect samples
  samples = [str(transform(r)) for _ in range(SAMPLES)]
  samples_data = '\n'.join(samples) + '\n'

  # Prepare command
  cmd = [BIN, '-b', str(bits), '-c', str(PREDICT)]
  print(f'Running: {" ".join(cmd)}')

  # Time the predictor invocation
  start = time.perf_counter()
  try:
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout, stderr = proc.communicate(input=samples_data)
    end = time.perf_counter()

    if proc.returncode != 0:
      print(f'Predictor failed after {end-start:.3f}s:')
      print(stderr)
      return False

  except Exception as e:
    end = time.perf_counter()
    print(f'Predictor failed after {end-start:.3f}s:')
    print(str(e))
    return False

  print(f'Invocation took {end - start:.3f} seconds')

  # Parse predictions
  predicted = list(map(int, stdout.strip().split()))
  # Generate actual next values
  actual = [transform(r) for _ in range(PREDICT)]

  # Compare
  if predicted == actual:
    print('Success: predictions match actual values')
    return True
  else:
    print('Failure: mismatch')
    print('Predicted:', predicted)
    print('Actual:   ', actual)
    return False


def main():
  all_ok = True
  for label, bits, transform in SCENARIOS:
    ok = run_scenario(label, bits, transform)
    all_ok &= ok
  if not all_ok:
    sys.exit(1)


if __name__ == '__main__':
  main()
