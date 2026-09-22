#!/usr/bin/env python3
"""Export recorded QPSO runs as Leaflet-ready JSON.

The exporter deliberately consumes completed JSONL logs; it never starts SUMO.
That makes the resulting files safe to bundle with the frontend.  Coordinates
are nevertheless derived from the SUMO network with sumolib's documented
``net.convertXY2LonLat(x, y)`` conversion--network coordinates are projected
metres, not longitude/latitude.

Examples
--------
Export one recorded run (the completion time is required when the raw log did
not record it)::

    python3 export_for_frontend.py export \
      --log logs/va_high_seed42.jsonl --net networks/delhi/delhi_intersection.net.xml \
      --scenario high --algorithm va_qpso --seed 42 --completion-time 842 \
      --output frontend/data/high_seed42_va_qpso.json

Export genuine matched pairs and select up to one winning adaptive run per
scenario tier::

    python3 export_for_frontend.py bundle --manifest frontend_runs.json \
      --net networks/delhi/delhi_intersection.net.xml --output-dir frontend/data

``frontend_runs.json`` contains ``{\"runs\": [...]}``, with each item having
``scenario``, ``seed``, ``va_log``, ``fixed_log``, ``va_completion_time`` and
``fixed_completion_time``.  A pair is rejected unless scenario and seed match.
"""

from __future__ import annotations

import argparse
import json
import math
from bisect import bisect_right
from pathlib import Path
from typing import Any, Iterable


ALGORITHMS = {"va_qpso", "fixed_beta_qpso"}


def require_sumolib():
    """Import lazily so ``--help`` works on a frontend-only machine."""
    try:
        import sumolib  # SUMO tools package; intentionally not replaced by XML math.
    except ImportError as exc:
        raise SystemExit(
            "sumolib is required to export coordinates. Install SUMO (or its "
            "Python tools) and retry; viewing already exported JSON needs no SUMO."
        ) from exc
    return sumolib


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number} is not valid JSON") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number} must contain a JSON object")
            events.append(record)
    if not events:
        raise ValueError(f"{path} has no events")
    return events


def validate_embedded_metadata(
    events: list[dict[str, Any]], scenario: str, algorithm: str, seed: int
) -> None:
    """Reject contradictory metadata while still supporting Day-5 legacy logs."""
    expected = {"scenario": scenario, "algorithm": algorithm, "seed": seed}
    for key, value in expected.items():
        recorded = {event[key] for event in events if key in event}
        if len(recorded) > 1:
            raise ValueError(f"The log contains conflicting {key!r} values: {sorted(recorded)!r}")
        if recorded and next(iter(recorded)) != value:
            raise ValueError(
                f"The log says {key}={next(iter(recorded))!r}, but the export requested {value!r}."
            )


def tier_for(volatility: float) -> str:
    if volatility < 0.25:
        return "calm"
    if volatility < 0.60:
        return "changing"
    return "volatile"


def event_detail(event: dict[str, Any]) -> str:
    kind = event.get("event")
    volatility = float(event.get("volatility_index", 0.0))
    if kind == "reroute":
        return "Took a clearer nearby road to avoid slow traffic."
    if event.get("trigger") == "arbiter":
        return "Several quick detours were needed, so the route was checked again."
    if volatility >= 0.60:
        return "Traffic is changing quickly, so the route was planned again."
    if volatility >= 0.25:
        return "Traffic is becoming less predictable, so the route was planned again."
    return "Traffic is steady, so the route was refreshed."


def frontend_events(events: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "t": round(float(event.get("sim_time", 0.0)), 2),
            "type": str(event.get("event", "update")),
            "detail": event_detail(event),
        }
        for event in events
        if event.get("event") in {"replan", "reroute"}
    ]


def latest_route_plan(events: list[dict[str, Any]]) -> tuple[list[str], list[int]]:
    plans = [event for event in events if event.get("event") == "replan"]
    if not plans:
        raise ValueError("The log has no replan event with stops and best_order.")
    plan = plans[-1]
    stops, order = plan.get("stops"), plan.get("best_order")
    if not isinstance(stops, list) or not isinstance(order, list):
        raise ValueError("The latest replan event is missing stops or best_order.")
    if sorted(order) != list(range(len(stops))):
        raise ValueError("best_order is not a permutation of the logged stops.")
    return [str(stop) for stop in stops], [int(index) for index in order]


def lonlat(net: Any, xy: tuple[float, float]) -> tuple[float, float]:
    """Return (lat, lon), always through sumolib's network conversion."""
    lon, lat = net.convertXY2LonLat(*xy)
    return float(lat), float(lon)


def stop_records(net: Any, stops: list[str]) -> list[dict[str, Any]]:
    records = []
    for number, stop_id in enumerate(stops, start=1):
        node = net.getNode(stop_id)
        if node is None:
            raise ValueError(f"Stop {stop_id!r} is not a junction in the network.")
        lat, lon = lonlat(net, node.getCoord())
        records.append({"id": stop_id, "lat": lat, "lon": lon, "label": f"Stop {number}"})
    return records


def shortest_edges_between_nodes(net: Any, from_node: str, to_node: str) -> list[Any]:
    """Find the shortest legal edge sequence between two logged junctions."""
    start = net.getNode(from_node)
    target = net.getNode(to_node)
    def candidates_for(vehicle_class: str | None) -> list[tuple[float, list[Any]]]:
        candidates: list[tuple[float, list[Any]]] = []
        for from_edge in start.getOutgoing():
            for to_edge in target.getIncoming():
                result = net.getShortestPath(from_edge, to_edge, vClass=vehicle_class)
                if result and result[0]:
                    edges, cost = result
                    candidates.append((float(cost), list(edges)))
        return candidates

    candidates = candidates_for("passenger")
    # Day-5's stop picker uses the graph topology without vehicle-class
    # filtering. Preserve those legacy recorded tours when an edge lacks an
    # explicit passenger permission, rather than silently dropping the run.
    if not candidates:
        candidates = candidates_for(None)
    if not candidates:
        raise ValueError(f"No network route from stop {from_node!r} to {to_node!r}.")
    return min(candidates, key=lambda candidate: candidate[0])[1]


def route_shapes(net: Any, ordered_stops: list[str]) -> tuple[list[tuple[float, float]], bool]:
    """Return edge-shape points; flag a legacy stop-geometry fallback if needed."""
    route: list[tuple[float, float]] = []
    used_stop_geometry_fallback = False
    for origin, destination in zip(ordered_stops, ordered_stops[1:]):
        try:
            edges = shortest_edges_between_nodes(net, origin, destination)
        except ValueError:
            # The only shipped Day-5 log records a TSP stop order with a
            # one-way-disconnected leg. It contains no actual vehicle edge
            # trace to recover. Keep the replay usable, but expose this fact
            # in path_source instead of pretending it is a road polyline.
            used_stop_geometry_fallback = True
            edges = []
            for xy in (net.getNode(origin).getCoord(), net.getNode(destination).getCoord()):
                point = lonlat(net, xy)
                if not route or point != route[-1]:
                    route.append(point)
        for edge in edges:
            for xy in edge.getShape():
                point = lonlat(net, xy)  # Required conversion for every edge-shape coordinate.
                if not route or point != route[-1]:
                    route.append(point)
    if len(route) < 2:
        raise ValueError("The reconstructed route has fewer than two shape coordinates.")
    return route, used_stop_geometry_fallback


def route_shapes_from_edges(net: Any, edge_ids: list[str]) -> list[tuple[float, float]]:
    """Use an exact edge route when a newer logger recorded one."""
    route: list[tuple[float, float]] = []
    for edge_id in edge_ids:
        edge = net.getEdge(edge_id)
        if edge is None:
            raise ValueError(f"Logged route references unknown edge {edge_id!r}.")
        for xy in edge.getShape():
            point = lonlat(net, xy)
            if not route or point != route[-1]:
                route.append(point)
    if len(route) < 2:
        raise ValueError("The logged edge route has fewer than two shape coordinates.")
    return route


def dense_path(points: list[tuple[float, float]], completion_time: float) -> list[dict[str, float]]:
    """Sample the polyline at roughly one visual point per simulated second."""
    if completion_time <= 0:
        raise ValueError("completion_time must be greater than zero.")
    # Equirectangular distance is sufficient for distributing timestamps across
    # this small city network; map coordinates themselves remain exact SUMO conversion.
    meters_per_degree = 111_320.0
    segment_lengths = []
    for (lat1, lon1), (lat2, lon2) in zip(points, points[1:]):
        dx = (lon2 - lon1) * meters_per_degree * math.cos(math.radians((lat1 + lat2) / 2))
        dy = (lat2 - lat1) * meters_per_degree
        segment_lengths.append(math.hypot(dx, dy))
    total = sum(segment_lengths)
    if total == 0:
        raise ValueError("The route shape has zero length.")

    sample_count = max(2, int(math.ceil(completion_time)) + 1)
    cumulative = [0.0]
    for length in segment_lengths:
        cumulative.append(cumulative[-1] + length)
    path = []
    for sample in range(sample_count):
        t = completion_time * sample / (sample_count - 1)
        distance = total * sample / (sample_count - 1)
        index = min(bisect_right(cumulative, distance) - 1, len(points) - 2)
        span = segment_lengths[index]
        fraction = (distance - cumulative[index]) / span if span else 0.0
        lat1, lon1 = points[index]
        lat2, lon2 = points[index + 1]
        path.append({"t": round(t, 2), "lat": lat1 + (lat2 - lat1) * fraction, "lon": lon1 + (lon2 - lon1) * fraction})
    return path


def metrics(events: list[dict[str, Any]], completion_time: float, algorithm: str) -> list[dict[str, Any]]:
    updates = sorted(
        (event for event in events if "volatility_index" in event),
        key=lambda event: float(event.get("sim_time", 0.0)),
    )
    if not updates:
        updates = [{"sim_time": 0.0, "volatility_index": 0.0}]
    result = []
    for event in updates:
        volatility = max(0.0, min(1.0, float(event["volatility_index"])))
        # fixed_beta_qpso anneals internally during each optimizer call. 0.75
        # is its midpoint, not a claim that it reacts to traffic.
        beta = 0.5 + 0.5 * volatility if algorithm == "va_qpso" else 0.75
        result.append({"t": round(min(float(event.get("sim_time", 0.0)), completion_time), 2), "volatility_index": volatility, "beta": beta, "tier": tier_for(volatility)})
    return result


def export_run(log: Path, net_file: Path, scenario: str, algorithm: str, seed: int, completion_time: float) -> dict[str, Any]:
    if algorithm not in ALGORITHMS:
        raise ValueError(f"algorithm must be one of {sorted(ALGORITHMS)}")
    sumolib = require_sumolib()
    events = read_jsonl(log)
    validate_embedded_metadata(events, scenario, algorithm, seed)
    stops, order = latest_route_plan(events)
    net = sumolib.net.readNet(str(net_file))
    ordered_stops = [stops[index] for index in order]
    # Day-5 logs contain stop order only, so reconstruct their edge route. A
    # richer logger may instead supply route_edges on its final replan event;
    # then preserve that exact recorded sequence.
    final_plan = [event for event in events if event.get("event") == "replan"][-1]
    exact_edges = final_plan.get("route_edges")
    if isinstance(exact_edges, list) and exact_edges:
        shapes = route_shapes_from_edges(net, [str(edge) for edge in exact_edges])
        path_source = "logged_network_edges"
    else:
        shapes, used_fallback = route_shapes(net, ordered_stops)
        path_source = "legacy_stop_geometry_fallback" if used_fallback else "reconstructed_network_edges"
    return {
        "scenario": scenario,
        "algorithm": algorithm,
        "seed": seed,
        "stops": stop_records(net, stops),
        "path": dense_path(shapes, completion_time),
        "path_source": path_source,
        "metrics_over_time": metrics(events, completion_time, algorithm),
        "events": frontend_events(events),
        "completion_time": completion_time,
    }


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def run_export(args: argparse.Namespace) -> None:
    data = export_run(Path(args.log), Path(args.net), args.scenario, args.algorithm, args.seed, args.completion_time)
    write_json(Path(args.output), data)
    print(f"Exported {args.output}")


def run_bundle(args: argparse.Namespace) -> None:
    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    runs = manifest.get("runs")
    if not isinstance(runs, list) or not runs:
        raise ValueError("manifest must contain a non-empty 'runs' list")
    output_dir = Path(args.output_dir)
    exported: list[tuple[float, str, Path]] = []
    for entry in runs:
        required = {"scenario", "seed", "va_log", "fixed_log", "va_completion_time", "fixed_completion_time"}
        missing = required - set(entry)
        if missing:
            raise ValueError(f"manifest entry is missing: {', '.join(sorted(missing))}")
        scenario, seed = str(entry["scenario"]), int(entry["seed"])
        va = export_run(manifest_path.parent / entry["va_log"], Path(args.net), scenario, "va_qpso", seed, float(entry["va_completion_time"]))
        fixed = export_run(manifest_path.parent / entry["fixed_log"], Path(args.net), scenario, "fixed_beta_qpso", seed, float(entry["fixed_completion_time"]))
        va_path = output_dir / f"{scenario}_seed{seed}_va_qpso.json"
        fixed_path = output_dir / f"{scenario}_seed{seed}_fixed_beta_qpso.json"
        write_json(va_path, va)
        write_json(fixed_path, fixed)
        improvement = (fixed["completion_time"] - va["completion_time"]) / fixed["completion_time"]
        if improvement > 0:
            exported.append((improvement, scenario, va_path))

    # Prefer one winner from each tier, then fill remaining slots by margin.
    chosen: list[Path] = []
    seen_scenarios: set[str] = set()
    for _, scenario, path in sorted(exported, reverse=True):
        if scenario not in seen_scenarios and len(chosen) < args.hero_count:
            chosen.append(path)
            seen_scenarios.add(scenario)
    for _, _, path in sorted(exported, reverse=True):
        if path not in chosen and len(chosen) < args.hero_count:
            chosen.append(path)
    write_json(output_dir / "hero_scenarios.json", {"heroes": [path.name for path in chosen]})
    print(f"Exported matched pairs to {output_dir}; selected {len(chosen)} verified hero scenario(s).")


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="Convert logged SUMO routes into frontend map data")
    commands = root.add_subparsers(dest="command", required=True)
    single = commands.add_parser("export", help="export one recorded run")
    single.add_argument("--log", required=True)
    single.add_argument("--net", required=True)
    single.add_argument("--scenario", required=True)
    single.add_argument("--algorithm", required=True, choices=sorted(ALGORITHMS))
    single.add_argument("--seed", required=True, type=int)
    single.add_argument("--completion-time", required=True, type=float)
    single.add_argument("--output", required=True)
    single.set_defaults(func=run_export)
    bundle = commands.add_parser("bundle", help="export matched pairs and choose verified heroes")
    bundle.add_argument("--manifest", required=True)
    bundle.add_argument("--net", required=True)
    bundle.add_argument("--output-dir", required=True)
    bundle.add_argument("--hero-count", type=int, default=3, choices=(2, 3))
    bundle.set_defaults(func=run_bundle)
    return root


if __name__ == "__main__":
    arguments = parser().parse_args()
    try:
        arguments.func(arguments)
    except (ValueError, OSError) as exc:
        raise SystemExit(f"Export failed: {exc}") from exc
