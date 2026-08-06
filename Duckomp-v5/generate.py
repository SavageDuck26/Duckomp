#!/usr/bin/env python3
"""
Generates sample.txt using the same in-house RNG as the Rust bruterforer.
Offsets (step sizes) are random values in the range [low, high].
Use same --offset-low / --offset-high values that the Rust bruterforer will compute
from the diffs (min diff, max diff).
"""

import struct


def rng_next(state: int) -> int:
    a = 6364136223846793005
    c = 1442695040888963407
    return (a * state + c) & 0xFFFFFFFFFFFFFFFF


def rng_range(state: int, low: int, high: int) -> tuple[int, int]:
    """Return (rng_value, new_state)"""
    state = rng_next(state)
    range_size = high - low + 1
    value = low + (state % range_size)
    return value, state


def generate(seed: int, offset_low: int, offset_high: int, count: int) -> list[int]:
    values = [1]
    state = seed
    for _ in range(count - 1):
        step, state = rng_range(state, offset_low, offset_high)
        values.append(values[-1] + step)
    return values


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate sample.txt with RNG-based data"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=12345,
        help="Seed for the RNG (default: 12345)",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=10,
        help="Number of values to generate (default: 10)",
    )
    parser.add_argument(
        "--offset-low",
        type=int,
        default=1,
        help="Minimum step size (default: 1)",
    )
    parser.add_argument(
        "--offset-high",
        type=int,
        default=50000,
        help="Maximum step size (default: 50000)",
    )
    parser.add_argument(
        "--output",
        default="sample.txt",
        help="Output file path (default: sample.txt)",
    )

    args = parser.parse_args()


    values = generate(args.seed, args.offset_low, args.offset_high, args.count)

    with open(args.output, "w") as f:
        for v in values:
            f.write(f"{v}\n")

    print(f"Generated {args.count} values with seed={args.seed}")
    print(f"Offsets: [{args.offset_low}, {args.offset_high}]")
    print(f"Values: {values[:5]}...{values[-3:]}" if len(values) > 8 else f"Values: {values}")
    print(f"Written to: {args.output}")