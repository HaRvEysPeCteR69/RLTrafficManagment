#!/usr/bin/env python3
"""
Test script to run and evaluate Low, Medium, and High Volatility traffic scenarios.

Runs each tier in SUMO using duarouter-compiled routes (dumb placeholder router),
collects performance telemetry from tripinfo and summary outputs, and displays
sanity statistics (average speed, vehicle count, waiting time, errors).
"""

import os
import sys
import argparse
import subprocess
import time
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Any, List

# Locate SUMO_HOME
if "SUMO_HOME" not in os.environ:
    possible_paths = [
        r"C:\Program Files (x86)\Eclipse\Sumo",
        r"C:\Program Files\Eclipse\Sumo",
        r"C:\Sumo",
        "/usr/share/sumo",
        "/opt/sumo",
    ]
    for p in possible_paths:
        if os.path.exists(p):
            os.environ["SUMO_HOME"] = p
            break

SUMO_HOME = os.environ.get("SUMO_HOME", "")


def get_binary(name: str) -> str:
    """Find SUMO executable."""
    exe_name = f"{name}.exe" if sys.platform == "win32" else name
    if SUMO_HOME:
        candidate = os.path.join(SUMO_HOME, "bin", exe_name)
        if os.path.exists(candidate):
            return candidate
    found = shutil.which(name)
    if found:
        return found
    raise FileNotFoundError(f"Binary {name} not found. Ensure SUMO is installed.")


def parse_simulation_results(
    tripinfo_path: str,
    summary_path: str,
    stdout: str,
    stderr: str,
    real_duration: float
) -> Dict[str, Any]:
    """Extract statistics from SUMO output XML files."""
    stats = {
        "vehicles_inserted": 0,
        "vehicles_arrived": 0,
        "vehicles_running": 0,
        "mean_speed_mps": 0.0,
        "mean_speed_kmh": 0.0,
        "mean_travel_time_s": 0.0,
        "mean_waiting_time_s": 0.0,
        "mean_time_loss_s": 0.0,
        "mean_route_length_m": 0.0,
        "teleports": 0,
        "real_time_s": round(real_duration, 2),
        "errors": []
    }

    # Check stderr for errors
    if stderr:
        for line in stderr.splitlines():
            line_str = line.strip()
            if "Error" in line_str or "Emergency" in line_str:
                stats["errors"].append(line_str)

    # Parse tripinfo
    if os.path.exists(tripinfo_path) and os.path.getsize(tripinfo_path) > 50:
        tree = ET.parse(tripinfo_path)
        root = tree.getroot()
        tripinfos = root.findall("tripinfo")

        if tripinfos:
            durations = [float(t.get("duration", 0)) for t in tripinfos]
            lengths = [float(t.get("routeLength", 0)) for t in tripinfos]
            waiting = [float(t.get("waitingTime", 0)) for t in tripinfos]
            loss = [float(t.get("timeLoss", 0)) for t in tripinfos]

            stats["vehicles_arrived"] = len(tripinfos)
            stats["mean_travel_time_s"] = round(sum(durations) / len(durations), 2)
            stats["mean_route_length_m"] = round(sum(lengths) / len(lengths), 2)
            stats["mean_waiting_time_s"] = round(sum(waiting) / len(waiting), 2)
            stats["mean_time_loss_s"] = round(sum(loss) / len(loss), 2)

            if sum(durations) > 0:
                stats["mean_speed_mps"] = round(sum(lengths) / sum(durations), 2)
                stats["mean_speed_kmh"] = round(stats["mean_speed_mps"] * 3.6, 2)

    # Parse summary (end of run snapshot)
    if os.path.exists(summary_path) and os.path.getsize(summary_path) > 50:
        tree = ET.parse(summary_path)
        root = tree.getroot()
        steps = root.findall("step")
        if steps:
            last_step = steps[-1]
            stats["vehicles_inserted"] = int(last_step.get("inserted", 0))
            stats["vehicles_running"] = int(last_step.get("running", 0))
            stats["teleports"] = int(last_step.get("teleports", 0))

            # If no vehicles finished yet, use summary's meanSpeed
            if stats["vehicles_arrived"] == 0 and float(last_step.get("meanSpeed", 0)) > 0:
                stats["mean_speed_mps"] = round(float(last_step.get("meanSpeed", 0)), 2)
                stats["mean_speed_kmh"] = round(stats["mean_speed_mps"] * 3.6, 2)

    return stats


def run_scenario_tier(
    tier: str,
    scenarios_dir: str,
    outputs_dir: str,
    duration: int,
    seed: int,
    use_gui: bool = False
) -> Dict[str, Any]:
    """Run a single volatility tier scenario in SUMO and return statistics."""
    tier_cfg = os.path.join(scenarios_dir, tier, "scenario.sumocfg")
    if not os.path.exists(tier_cfg):
        raise FileNotFoundError(f"Scenario configuration not found: {tier_cfg}. Run generate_scenarios.py first.")

    os.makedirs(outputs_dir, exist_ok=True)
    tripinfo_file = os.path.join(outputs_dir, f"{tier}_tripinfo.xml")
    summary_file = os.path.join(outputs_dir, f"{tier}_summary.xml")

    sumo_bin = get_binary("sumo-gui" if use_gui else "sumo")

    cmd = [
        sumo_bin,
        "-c", tier_cfg,
        "--tripinfo-output", tripinfo_file,
        "--summary-output", summary_file,
        "--seed", str(seed),
        "--end", str(duration),
        "--ignore-route-errors", "true",
        "--device.rerouting.probability", "1.0",
        "--device.rerouting.adaptation-interval", "5",
        "--no-step-log", "true",
        "--no-warnings", "true",
    ]
    if use_gui:
        cmd.extend(["--start", "--quit-on-end"])

    t0 = time.time()
    res = subprocess.run(cmd, capture_output=True, text=True)
    real_time = time.time() - t0

    stats = parse_simulation_results(
        tripinfo_path=tripinfo_file,
        summary_path=summary_file,
        stdout=res.stdout,
        stderr=res.stderr,
        real_duration=real_time
    )
    stats["tier"] = tier.upper()
    stats["exit_code"] = res.returncode
    return stats


def print_comparison_table(results: List[Dict[str, Any]]):
    """Format and print comparative results in a clean table."""
    print("\n" + "=" * 95)
    print("DELHI NETWORK SCENARIO BENCHMARK (DUAROUTER PLACEHOLDER)")
    print("=" * 95)

    headers = [
        "Tier", "Inserted", "Arrived", "Running", "Avg Speed (km/h)",
        "Avg Trip (s)", "Avg Wait (s)", "Teleports", "Sim Runtime", "Status"
    ]
    row_fmt = "{:<8} {:>9} {:>8} {:>8} {:>17} {:>13} {:>13} {:>10} {:>12} {:>8}"
    print(row_fmt.format(*headers))
    print("-" * 95)

    for r in results:
        status = "OK" if r["exit_code"] == 0 and len(r["errors"]) == 0 else "WARN"
        print(row_fmt.format(
            r["tier"],
            r["vehicles_inserted"],
            r["vehicles_arrived"],
            r["vehicles_running"],
            f"{r['mean_speed_kmh']} km/h",
            f"{r['mean_travel_time_s']}s",
            f"{r['mean_waiting_time_s']}s",
            r["teleports"],
            f"{r['real_time_s']}s",
            status
        ))

    print("=" * 95)


def main():
    parser = argparse.ArgumentParser(description="Test Delhi Low/Medium/High Volatility Scenarios")
    parser.add_argument("--tiers", nargs="+", default=["low", "medium", "high"], help="Tiers to run")
    parser.add_argument("--duration", type=int, default=300, help="Simulation duration (default 300s = 5m)")
    parser.add_argument("--seed", type=int, default=42, help="Seed for simulation run")
    parser.add_argument("--regenerate", action="store_true", help="Re-generate scenarios before running")
    parser.add_argument("--gui", action="store_true", help="Run with SUMO-GUI")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    scenarios_dir = str(project_root / "networks" / "delhi" / "scenarios")
    outputs_dir = str(project_root / "outputs" / "scenario_tests")

    if args.regenerate:
        print("Regenerating scenarios with seed:", args.seed)
        gen_script = str(project_root / "networks" / "delhi" / "generate_scenarios.py")
        cmd = [sys.executable, gen_script, "--tier", "all", "--duration", str(args.duration), "--seed", str(args.seed)]
        subprocess.run(cmd, check=True)

    results = []
    for tier in args.tiers:
        print(f"Running simulation for Tier [{tier.upper()}] (duration: {args.duration}s, seed: {args.seed})...")
        res = run_scenario_tier(
            tier=tier.lower(),
            scenarios_dir=scenarios_dir,
            outputs_dir=outputs_dir,
            duration=args.duration,
            seed=args.seed,
            use_gui=args.gui
        )
        results.append(res)

    print_comparison_table(results)

    # Sanity checks
    print("\nSanity Check Assertions:")
    all_passed = True
    for r in results:
        tier = r["tier"]
        if r["vehicles_inserted"] > 0:
            print(f"  [PASS] {tier}: Vehicles loaded and inserted successfully ({r['vehicles_inserted']} total).")
        else:
            print(f"  [FAIL] {tier}: No vehicles inserted.")
            all_passed = False

        if r["mean_speed_kmh"] > 0 or r["mean_speed_mps"] > 0:
            print(f"  [PASS] {tier}: Positive vehicle motion recorded ({r['mean_speed_kmh']} km/h).")
        else:
            print(f"  [FAIL] {tier}: Vehicles failed to move.")
            all_passed = False

        if r["exit_code"] == 0:
            print(f"  [PASS] {tier}: Simulation completed with exit code 0 (no fatal crashes).")
        else:
            print(f"  [FAIL] {tier}: Exited with non-zero code {r['exit_code']}.")
            all_passed = False

    if all_passed:
        print("\nAll sanity checks PASSED! The traffic world runs smoothly across all volatility tiers.")
    else:
        print("\nSome sanity checks FAILED. Review errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
