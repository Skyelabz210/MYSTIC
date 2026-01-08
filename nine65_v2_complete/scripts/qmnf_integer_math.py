#!/usr/bin/env python3
"""
QMNF Integer Math Utilities

Integer-only mathematical functions for QMNF compliance.
Replaces float-based math operations with exact integer arithmetic.

All functions use scaled integers to maintain precision without floats.
Default scale factor: 10^6 (SCALE = 1_000_000)

Author: QMNF System | January 2026
"""

from typing import Tuple

# Default scale factor for fixed-point arithmetic
SCALE = 1_000_000  # 10^6

# Maximum integer value (replaces float('inf'))
INT_MAX = (1 << 63) - 1
INT_MIN = -INT_MAX


def isqrt(n: int) -> int:
    """
    Integer square root using Newton's method.

    Returns floor(sqrt(n)) exactly.

    Args:
        n: Non-negative integer

    Returns:
        Largest integer k such that k*k <= n

    Example:
        isqrt(10) = 3  (since 3*3=9 <= 10 < 16=4*4)
        isqrt(100) = 10
    """
    if n < 0:
        raise ValueError("Square root of negative number")
    if n < 2:
        return n

    # Initial estimate
    x = n
    y = (x + 1) // 2

    # Newton-Raphson iteration
    while y < x:
        x = y
        y = (x + n // x) // 2

    return x


def isqrt_scaled(n_scaled: int, scale: int = SCALE) -> int:
    """
    Integer square root for scaled values.

    If n_scaled = n * scale, returns sqrt(n) * sqrt(scale).

    Args:
        n_scaled: Value multiplied by scale factor
        scale: Scale factor used

    Returns:
        sqrt(n_scaled) as integer
    """
    return isqrt(n_scaled)


def distance_squared_3d(x1: int, y1: int, z1: int, x2: int, y2: int, z2: int) -> int:
    """
    Compute squared Euclidean distance in 3D space.

    Avoids sqrt by returning distance squared.
    For comparison purposes, d1 < d2 iff d1^2 < d2^2.

    Args:
        x1, y1, z1: First point coordinates
        x2, y2, z2: Second point coordinates

    Returns:
        (x2-x1)^2 + (y2-y1)^2 + (z2-z1)^2
    """
    dx = x2 - x1
    dy = y2 - y1
    dz = z2 - z1
    return dx * dx + dy * dy + dz * dz


def distance_3d_scaled(x1: int, y1: int, z1: int,
                       x2: int, y2: int, z2: int,
                       scale: int = SCALE) -> int:
    """
    Compute Euclidean distance in 3D space with scaled coordinates.

    All coordinates should be pre-scaled by 'scale' factor.
    Returns distance scaled by the same factor.

    Args:
        x1, y1, z1: First point coordinates (scaled)
        x2, y2, z2: Second point coordinates (scaled)
        scale: Scale factor used for coordinates

    Returns:
        Distance as scaled integer
    """
    dist_sq = distance_squared_3d(x1, y1, z1, x2, y2, z2)
    # sqrt(dist_sq) where dist_sq is in units of scale^2
    # Result should be in units of scale
    return isqrt(dist_sq)


def ellipsoid_distance_scaled(point: Tuple[int, int, int],
                               center: Tuple[int, int, int],
                               radii: Tuple[int, int, int],
                               scale: int = SCALE) -> int:
    """
    Compute normalized ellipsoid distance.

    Returns distance normalized by radii, scaled by SCALE.
    A value of SCALE means the point is on the ellipsoid boundary.
    Less than SCALE means inside, greater means outside.

    Args:
        point: (x, y, z) coordinates (scaled)
        center: (cx, cy, cz) ellipsoid center (scaled)
        radii: (rx, ry, rz) ellipsoid radii (scaled)
        scale: Scale factor

    Returns:
        Normalized distance * scale
    """
    dx = point[0] - center[0]
    dy = point[1] - center[1]
    dz = point[2] - center[2]

    # Normalize by radii: (dx/rx)^2 + (dy/ry)^2 + (dz/rz)^2
    # To avoid division, multiply out: dx^2 * ry^2 * rz^2 + dy^2 * rx^2 * rz^2 + dz^2 * rx^2 * ry^2
    # Then divide by (rx * ry * rz)^2

    rx, ry, rz = radii

    # Scale-adjusted calculation
    term_x = (dx * dx * scale * scale) // (rx * rx) if rx != 0 else 0
    term_y = (dy * dy * scale * scale) // (ry * ry) if ry != 0 else 0
    term_z = (dz * dz * scale * scale) // (rz * rz) if rz != 0 else 0

    dist_sq_scaled = term_x + term_y + term_z

    # Return sqrt of sum
    return isqrt(dist_sq_scaled)


def membership_score(point: Tuple[int, int, int],
                     center: Tuple[int, int, int],
                     radii: Tuple[int, int, int],
                     scale: int = SCALE) -> int:
    """
    Calculate basin membership score (0 to scale).

    Returns:
        scale (1.0) at center
        0 at boundary
        Negative values outside (clamped to 0)

    This replaces the float-based point_in_basin function.
    """
    dist = ellipsoid_distance_scaled(point, center, radii, scale)
    membership = scale - dist
    return max(0, membership)


def div_scaled(a: int, b: int, scale: int = SCALE) -> int:
    """
    Integer division with scaling to preserve precision.

    Computes (a / b) * scale without using floats.

    Args:
        a: Numerator
        b: Denominator (non-zero)
        scale: Scale factor for result

    Returns:
        (a * scale) // b
    """
    if b == 0:
        raise ValueError("Division by zero")
    return (a * scale) // b


def mul_scaled(a: int, b: int, scale: int = SCALE) -> int:
    """
    Multiply two scaled values and rescale.

    If a and b are both scaled by 'scale', their product
    would be scaled by scale^2. This rescales to scale.

    Args:
        a: First scaled value
        b: Second scaled value
        scale: Scale factor

    Returns:
        (a * b) // scale
    """
    return (a * b) // scale


def abs_int(n: int) -> int:
    """Integer absolute value."""
    return n if n >= 0 else -n


def clamp(value: int, min_val: int, max_val: int) -> int:
    """Clamp value to range [min_val, max_val]."""
    if value < min_val:
        return min_val
    if value > max_val:
        return max_val
    return value


def min_int(*args: int) -> int:
    """Find minimum of integer arguments."""
    result = args[0]
    for a in args[1:]:
        if a < result:
            result = a
    return result


def max_int(*args: int) -> int:
    """Find maximum of integer arguments."""
    result = args[0]
    for a in args[1:]:
        if a > result:
            result = a
    return result


# =============================================================================
# SELF-TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("QMNF INTEGER MATH UTILITIES - TEST SUITE")
    print("=" * 70)

    print("\n[Test 1] Integer Square Root")
    test_cases = [(0, 0), (1, 1), (4, 2), (9, 3), (10, 3), (100, 10),
                  (1000000, 1000), (2147483647, 46340)]
    for n, expected in test_cases:
        result = isqrt(n)
        status = "✓" if result == expected else "✗"
        print(f"  isqrt({n}) = {result} (expected {expected}) {status}")

    print("\n[Test 2] 3D Distance Squared")
    d_sq = distance_squared_3d(0, 0, 0, 3, 4, 0)
    expected_sq = 25  # 3^2 + 4^2 = 9 + 16 = 25
    print(f"  distance_squared(0,0,0 to 3,4,0) = {d_sq} (expected 25) {'✓' if d_sq == 25 else '✗'}")

    print("\n[Test 3] Scaled Division")
    result = div_scaled(1, 3, SCALE)
    expected = 333333  # 1/3 * 10^6
    print(f"  1/3 scaled = {result} (expected ~333333) {'✓' if abs(result - expected) < 2 else '✗'}")

    print("\n[Test 4] Membership Score")
    # Point at center should have full membership
    center = (0, 0, 0)
    radii = (SCALE, SCALE, SCALE)
    point_center = (0, 0, 0)
    point_boundary = (SCALE, 0, 0)
    point_outside = (2 * SCALE, 0, 0)

    score_center = membership_score(point_center, center, radii)
    score_boundary = membership_score(point_boundary, center, radii)
    score_outside = membership_score(point_outside, center, radii)

    print(f"  Center membership: {score_center} (expected {SCALE}) {'✓' if score_center == SCALE else '✗'}")
    print(f"  Boundary membership: {score_boundary} (expected 0) {'✓' if score_boundary == 0 else '✗'}")
    print(f"  Outside membership: {score_outside} (expected 0) {'✓' if score_outside == 0 else '✗'}")

    print("\n" + "=" * 70)
    print("✓ QMNF Integer Math Utilities Ready")
    print("=" * 70)
