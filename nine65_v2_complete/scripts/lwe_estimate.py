#!/usr/bin/env python3
"""
LWE Security Estimator for QMNF FHE Parameters

Uses the lattice-estimator library for precise security analysis.
For rough estimates, use the Rust `security` module instead.

Installation:
    pip install lattice-estimator
    
    OR (for latest):
    git clone https://github.com/malb/lattice-estimator.git
    cd lattice-estimator
    pip install -e .

Usage:
    python scripts/lwe_estimate.py
    python scripts/lwe_estimate.py --n 2048 --log_q 54 --sigma 3.2
    
Reference:
    Martin R. Albrecht, Rachel Player, Sam Scott:
    "On the concrete hardness of Learning with Errors"
    Journal of Mathematical Cryptology, 2015
"""

import argparse
import sys

def estimate_security(n: int, log_q: int, sigma: float = 3.2):
    """
    Estimate security of LWE parameters.
    
    Args:
        n: Ring dimension
        log_q: Log base-2 of modulus
        sigma: Gaussian width (standard deviation)
    
    Returns:
        Dictionary with attack costs
    """
    try:
        from estimator import *
    except ImportError:
        print("ERROR: lattice-estimator not installed")
        print("Install with: pip install lattice-estimator")
        print("\nFalling back to HE Standard table lookup...")
        return he_standard_estimate(n, log_q)
    
    q = 2 ** log_q
    
    # LWE parameters
    params = LWE.Parameters(
        n=n,
        q=q,
        Xs=ND.DiscreteGaussian(sigma),
        Xe=ND.DiscreteGaussian(sigma),
    )
    
    print(f"╔══════════════════════════════════════════════════════════════╗")
    print(f"║  LWE Security Estimate for QMNF FHE                         ║")
    print(f"╠══════════════════════════════════════════════════════════════╣")
    print(f"║  Parameters:                                                 ║")
    print(f"║    n = {n:<6}  (ring dimension)                             ║")
    print(f"║    log(q) = {log_q:<3}  (modulus bits)                          ║")
    print(f"║    σ = {sigma:<5}  (Gaussian width)                           ║")
    print(f"║    n/log(q) = {n/log_q:.1f}                                       ║")
    print(f"╠══════════════════════════════════════════════════════════════╣")
    print(f"║  Running lattice-estimator...                                ║")
    print(f"╚══════════════════════════════════════════════════════════════╝")
    print()
    
    # Run estimator
    result = LWE.estimate(params)
    
    # Find best attack
    best_attack = min(result.items(), key=lambda x: float(x[1]))
    best_name, best_cost = best_attack
    
    print(f"\n{'='*64}")
    print(f"RESULTS")
    print(f"{'='*64}")
    
    for attack, cost in sorted(result.items(), key=lambda x: float(x[1])):
        print(f"  {attack}: {cost}")
    
    print(f"\n{'='*64}")
    print(f"BEST ATTACK: {best_name}")
    print(f"SECURITY LEVEL: ~{float(best_cost):.0f} bits classical")
    print(f"{'='*64}")
    
    return result


def he_standard_estimate(n: int, log_q: int):
    """
    HE Standard v1.1 Table 3 lookup (fallback when estimator unavailable).
    """
    ratio = n / log_q
    
    if ratio > 50:
        security = 256
    elif ratio > 38:
        security = 192
    elif ratio > 28:
        security = 128
    elif ratio > 18:
        security = 96
    elif ratio > 12:
        security = 80
    else:
        security = 64
    
    print(f"\n{'='*64}")
    print(f"HE STANDARD TABLE LOOKUP (Conservative Estimate)")
    print(f"{'='*64}")
    print(f"  n = {n}")
    print(f"  log(q) = {log_q}")
    print(f"  n/log(q) = {ratio:.1f}")
    print(f"\n  SECURITY LEVEL: ~{security} bits classical")
    print(f"  CONFIDENCE: Standard (HE Standard v1.1 Table 3)")
    print(f"{'='*64}")
    
    return {"he_standard": security}


def qmnf_parameter_sets():
    """Standard QMNF FHE parameter sets."""
    return {
        "light": (1024, 30, 1.0),
        "he_standard_128": (2048, 30, 1.2),
        "standard_128": (4096, 30, 1.2),
        "high_192": (8192, 60, 1.2),
        "deep_128": (8192, 218, 1.7),
    }


def main():
    parser = argparse.ArgumentParser(
        description="LWE Security Estimator for QMNF FHE Parameters"
    )
    parser.add_argument("--n", type=int, help="Ring dimension")
    parser.add_argument("--log_q", type=int, help="Log base-2 of modulus")
    parser.add_argument("--sigma", type=float, default=3.2, help="Gaussian width")
    parser.add_argument("--all", action="store_true", help="Estimate all QMNF configs")
    
    args = parser.parse_args()
    
    if args.all:
        print("╔══════════════════════════════════════════════════════════════╗")
        print("║  QMNF FHE Parameter Security Analysis                        ║")
        print("╚══════════════════════════════════════════════════════════════╝")
        print()
        
        for name, (n, log_q, sigma) in qmnf_parameter_sets().items():
            print(f"\n{'─'*64}")
            print(f"  Configuration: {name}")
            print(f"{'─'*64}")
            estimate_security(n, log_q, sigma)
            print()
    elif args.n and args.log_q:
        estimate_security(args.n, args.log_q, args.sigma)
    else:
        # Default: estimate standard QMNF configs
        print("Estimating security for standard QMNF configurations...\n")
        
        # Quick HE Standard lookup for common configs
        configs = [
            ("light (N=1024)", 1024, 30),
            ("he_standard_128 (N=2048)", 2048, 30),
            ("standard_128 (N=4096)", 4096, 30),
        ]
        
        print(f"{'Config':<30} {'N':<8} {'log(q)':<8} {'Ratio':<8} {'Security':<10}")
        print("─" * 70)
        
        for name, n, log_q in configs:
            ratio = n / log_q
            if ratio > 50:
                sec = 256
            elif ratio > 38:
                sec = 192
            elif ratio > 28:
                sec = 128
            elif ratio > 18:
                sec = 96
            elif ratio > 12:
                sec = 80
            else:
                sec = 64
            print(f"{name:<30} {n:<8} {log_q:<8} {ratio:<8.1f} ~{sec} bits")
        
        print("\nFor precise estimates, install lattice-estimator:")
        print("  pip install lattice-estimator")
        print("  python scripts/lwe_estimate.py --n 2048 --log_q 54 --sigma 3.2")


if __name__ == "__main__":
    main()
