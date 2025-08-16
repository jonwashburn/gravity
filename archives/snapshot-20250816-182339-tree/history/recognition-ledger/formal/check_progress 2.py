#!/usr/bin/env python3
"""
Progress Checker for Recognition Science Solver
===============================================

Displays current status of proof generation without running the solver.
"""

import json
import os
from datetime import datetime
from pathlib import Path

def format_time(seconds):
    """Format seconds into human-readable time"""
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    return f"{hours:.0f}h {minutes:.0f}m"

def main():
    progress_file = Path("recognition_progress.json")
    
    if not progress_file.exists():
        print("❌ No progress file found. Run the solver first.")
        return
    
    with open(progress_file) as f:
        data = json.load(f)
    
    print("=" * 70)
    print("RECOGNITION SCIENCE PROOF PROGRESS")
    print("=" * 70)
    
    # Basic stats
    stats = data.get("statistics", {})
    proven_count = stats.get("proven_count", 0)
    total_theorems = len(data.get("theorems", {}))
    
    print(f"Last updated: {data.get('timestamp', 'Unknown')}")
    print(f"Proven: {proven_count}/{total_theorems} ({proven_count/total_theorems*100:.1f}%)")
    print(f"Runtime: {format_time(stats.get('runtime_seconds', 0))}")
    print(f"API calls: {stats.get('api_calls', 0):,}")
    print(f"Total tokens: {stats.get('total_tokens', 0):,}")
    print(f"Model escalations: {stats.get('model_escalations', 0)}")
    
    # Cost estimate
    estimated_cost = (stats.get('total_tokens', 0) / 1000) * 0.015
    print(f"Estimated cost: ${estimated_cost:.2f}")
    
    print("\n" + "-" * 70)
    print("PROGRESS BY LEVEL:")
    print("-" * 70)
    
    # Group by level
    level_stats = {}
    for name, theorem in data.get("theorems", {}).items():
        # Extract level from name
        if name.startswith("A"):
            level = 0
        elif name.startswith("F"):
            level = 1
        elif name.startswith("C"):
            level = 2
        elif name.startswith("E"):
            level = 3
        elif name.startswith("G"):
            level = 4
        elif name.startswith("P"):
            level = 5
        else:
            level = -1
        
        if level not in level_stats:
            level_stats[level] = {"total": 0, "proven": 0}
        
        level_stats[level]["total"] += 1
        if theorem.get("status") in ["proven", "given"]:
            level_stats[level]["proven"] += 1
    
    level_names = {
        0: "Axioms",
        1: "Foundation",
        2: "Core",
        3: "Energy Cascade",
        4: "Gauge Structure",
        5: "Predictions"
    }
    
    for level in sorted(level_stats.keys()):
        if level >= 0:
            stats = level_stats[level]
            name = level_names.get(level, f"Level {level}")
            proven = stats["proven"]
            total = stats["total"]
            percent = proven/total*100 if total > 0 else 0
            
            # Progress bar
            bar_length = 30
            filled = int(bar_length * proven / total) if total > 0 else 0
            bar = "█" * filled + "░" * (bar_length - filled)
            
            print(f"Level {level} - {name:15} [{bar}] {proven:2}/{total:2} ({percent:5.1f}%)")
    
    print("\n" + "-" * 70)
    print("CRITICAL THEOREMS:")
    print("-" * 70)
    
    critical = {
        "C1_GoldenRatioLockIn": "Golden Ratio Lock-In",
        "E1_CoherenceQuantum": "Coherence Quantum (0.090 eV)",
        "P1_ElectronMass": "Electron Mass Prediction",
        "P3_FineStructure": "Fine Structure Constant",
        "P5_DarkEnergy": "Dark Energy Density",
        "P6_HubbleConstant": "Hubble Constant"
    }
    
    for theorem_id, description in critical.items():
        theorem = data.get("theorems", {}).get(theorem_id, {})
        status = theorem.get("status", "unknown")
        attempts = theorem.get("attempts", 0)
        
        if status == "proven":
            status_icon = "✅"
        elif status == "given":
            status_icon = "📋"
        elif attempts > 0:
            status_icon = "🔄"
        else:
            status_icon = "⏳"
        
        print(f"{status_icon} {theorem_id}: {description}")
        print(f"   Status: {status} | Attempts: {attempts}")
    
    # Recent activity
    print("\n" + "-" * 70)
    print("UNPROVEN THEOREMS:")
    print("-" * 70)
    
    unproven = []
    for name, theorem in data.get("theorems", {}).items():
        if theorem.get("status") == "unproven":
            unproven.append((name, theorem.get("attempts", 0)))
    
    if unproven:
        unproven.sort(key=lambda x: x[1], reverse=True)
        print(f"Total unproven: {len(unproven)}")
        print("\nMost attempted:")
        for name, attempts in unproven[:5]:
            print(f"  - {name}: {attempts} attempts")
    else:
        print("🎉 All theorems proven!")
    
    print("\n" + "=" * 70)

if __name__ == "__main__":
    main() 