# Copyright 2026 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Runs XNNPACK's unit tests on the current machine for the standalone GN build.

Collects everything in the build output directory matching the pattern
out/foo/xnnpack_*_test, and runs them with GoogleTest sharding enabled.
"""

import argparse
import asyncio
import dataclasses
import datetime
import glob
import os
import sys
from typing import Any


@dataclasses.dataclass
class TestResult:
  stdout: str
  stderr: str
  suite: str
  success: bool
  duration_seconds: float
  shard: int


async def run_one_test(
    path_to_executable: os.PathLike,
    current_shard: int,
    total_shards: int,
    lock: asyncio.Semaphore,
) -> TestResult:
  """Runs a single test suite with the given shard index."""
  # Prepare arguments etc
  args = ['gtest_brief=1', 'gtest_color=0']
  env = {
      'GTEST_TOTAL_SHARDS': str(total_shards),
      'GTEST_SHARD_INDEX': str(current_shard),
  }
  async with lock:
    start_time = datetime.datetime.now()
    # Start the test process running
    process = await asyncio.create_subprocess_exec(
        path_to_executable,
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env,
    )

    # Pull back the stdout and stderr
    stdout, stderr = await process.communicate()
    await process.wait()
    end_time = datetime.datetime.now()
    duration = (end_time - start_time).total_seconds()

    # Await the exit code, if zero, all good
    return TestResult(
        stdout=stdout.decode('ascii'),
        stderr=stderr.decode('ascii'),
        suite=os.path.basename(path_to_executable),
        success=process.returncode == 0,
        duration_seconds=duration,
        shard=current_shard,
    )


async def main():
  parser = argparse.ArgumentParser(
      'run-gn-tests',
      description="Runs XNNPACK's unit tests on the"
      + ' current machine, and print a summary.',
  )
  parser.add_argument(
      'out_dir', help='Path to a build directory e.g. out/Default'
  )
  # 8 shards is overkill for some of the smaller tests
  parser.add_argument(
      '--shards',
      type=int,
      default=8,
      help='How much to subdivide each test suite',
  )
  parser.add_argument(
      '--cpus',
      type=int,
      help='The maximum number of test shards that can run at once',
  )
  parser.add_argument(
      '--verbose',
      action='store_true',
      help="Prints test output as they're executing",
  )

  args = parser.parse_args()

  # Figure out how many tests we can run at once, the command line
  # takes precendence.
  detected_concurrency = os.cpu_count()
  concurrency = (
      args.cpus
      if args.cpus
      else (1 if not detected_concurrency else detected_concurrency)
  )
  # The semaphore controls the number of tests that can run.
  semaphore = asyncio.Semaphore(concurrency)

  # Pick up the executables - must be named in this way to work
  test_suites = list(sorted(glob.glob(args.out_dir + '/xnnpack_*_test')))

  print(f'Discovered {len(test_suites)} test suites...')
  task_list = []
  for suite in test_suites:
    for shard in range(args.shards):
      task_list.append(run_one_test(suite, shard, args.shards, semaphore))

  failures = []
  for result in asyncio.as_completed(task_list):
    result = await result
    description = f'{result.suite} ({result.shard}/{args.shards})'
    print(description.ljust(60, '.'), end='', flush=True)
    outcome = (
        f'PASS ({result.duration_seconds:.2f} s)' if result.success else 'FAIL'
    )
    print(outcome.rjust(20, '.'))
    if args.verbose:
      print(result.stdout)
      if result.stderr:
        print('**stderr*')
        print(result.stderr)
    if not result.success:
      failures.append(result)

  # Re-iterate any failures
  for x in sorted(failures, key=lambda x: x.suite):
    print(x.suite, f'- Shard #{x.shard}', 'stderr:')
    print(x.stderr)
    print('stdout:')
    print(x.stdout)

  # Print a final summary and exit
  if not failures:
    print('** SUCCESS - ALL TESTS PASS **')
  else:
    print('** TEST FAILURES **')

  sys.exit(int(len(failures) >= 1))


if __name__ == '__main__':
  asyncio.run(main())
