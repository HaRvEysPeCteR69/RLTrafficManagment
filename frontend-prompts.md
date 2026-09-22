# Frontend Build Prompts — Judge-Ready, Explains-Itself Demo

## Design principles (read before prompting)

**"Anyone could use it"** means: no setup, no login, no technical background
needed. A judge opens a link, sees an intro screen in plain language, clicks
play, and understands what's happening without you narrating every second.

**"Highlights the project's working and logic"** means the UI doesn't just show
a route on a map — it makes the *adaptive mechanism visible*: the volatility
number changing, the system deciding to search harder or re-plan sooner because
of it, and a side-by-side against the fixed-schedule version so the judge can
see the difference with their own eyes, not just hear you claim it.

**Reliability beats polish.** SIH's own judging weight puts real emphasis on
the demo simply working — a crashed live simulation in front of judges is one
of the most common ways strong technical work still scores low. So: the primary
demo path replays real, precomputed logs from actual runs (not faked — genuinely
captured data from your experiments), not a live SUMO/TraCI connection. This is
still an honest demo of a real system; it's just not gambling your score on a
live socket connection working under pressure.

**Architecture:** a small FastAPI backend serving your logged run data, and a
single HTML/JS page (Leaflet for the map, no heavy framework needed) as the
frontend. Simple enough to build in the time you have, capable enough to look
like a real product.

---

## Prompt 1 — Bridge your SUMO data to something a browser can draw

```
Build export_for_frontend.py. Using sumolib (import sumolib; net =
sumolib.net.readNet(path_to_net_file)), for each edge in the route of a
logged run, get its shape coordinates and convert them from SUMO's internal
Cartesian system to lon/lat using net.convertXY2LonLat(x, y) — this is the
documented sumolib method for this exact conversion, needed because SUMO
networks are in projected meters and Leaflet/OpenStreetMap need lon/lat.

Take one of run_hybrid.py's JSON logs (from Day 5) and produce a
frontend-friendly JSON file containing:
{
  "scenario": "high_volatility",
  "algorithm": "va_qpso",
  "stops": [{"id": ..., "lat": ..., "lon": ..., "label": "Stop 1"}, ...],
  "path": [{"t": 0, "lat": ..., "lon": ...}, ...],   // vehicle position over time, densely sampled
  "metrics_over_time": [{"t": 0, "volatility_index": 0.1, "beta": 0.55, "tier": "calm"}, ...],
  "events": [{"t": 34, "type": "replan", "detail": "Traffic getting less predictable — re-planning route"},
             {"t": 61, "type": "reroute", "detail": "Detoured around a jam on MG Road"}],
  "completion_time": 842
}
Write the "detail" strings in plain, non-technical language — these get shown
directly to judges. Run this for both va_qpso and fixed_beta_qpso on the SAME
seed/scenario, so I have a matched pair to show side by side.

Then pick your best 2-3 "hero" scenarios (ones where the adaptive version
visibly wins, ideally one from each volatility tier) and export those — these
become the bundled, guaranteed-to-work demo data, independent of whether SUMO
even runs on the judging machine.
```


## Prompt 3 — The map and the animated vehicle

```
Build index.html: a single page using Leaflet (via CDN) with OpenStreetMap
tiles, centered on our network's bounding box. On load, fetch one run's JSON
from /api/runs/{id} and:
- Draw numbered pin markers for each stop.
- Draw the route as a polyline.
- Animate a vehicle marker moving along the "path" array using
  requestAnimationFrame, interpolating position based on elapsed playback
  time versus each path point's "t" value — keep this hand-written rather
  than pulling in an extra moving-marker plugin, it's a small amount of code
  and keeps the page dependency-light.
- Color the vehicle marker or route by the current tier (calm/medium/chaotic)
  so there's a visual cue even before a judge reads any numbers.
```

## Prompt 4 — Live metrics panel, explained in plain language

```
Add a metrics sidebar/topbar to index.html, updating as playback advances
through metrics_over_time:
- A volatility gauge: a simple semicircular or bar gauge from 0 to 1, colored
  green→yellow→red. Label it "Traffic unpredictability" not "volatility index"
  — add a small (i) info icon next to it that shows on hover/tap: "How much
  traffic conditions have been changing recently. Higher means more chaotic."
- The current tier as a badge (Calm / Moderate / Chaotic).
- Current search aggressiveness (beta), labeled "How hard the system is
  searching right now" with a similar info tooltip explaining that it rises
  automatically when traffic gets less predictable.
- Simulated clock (elapsed time).
Every technical term visible to the user needs a plain-language label as the
primary text, with the technical term available on hover for anyone who wants
it — never show only "β = 0.71" with no explanation.
```

## Prompt 5 — Event feed: make the decision-making visible

```
Add a scrolling event feed panel to index.html. As playback crosses each
timestamp in the "events" array, append that event's "detail" text to the
feed with a small icon (🔄 for replan, ↪️ for reroute) and the sim time. This
is the single most important panel for "highlighting the project's logic" —
a judge watching this feed should be able to narrate back to you what the
system just did and why, without you explaining it.
```

## Prompt 6 — Adaptive vs. Fixed, side by side

```
Add a comparison mode: a toggle or split-screen showing two synced playbacks
of the SAME seed/scenario — one using the va_qpso hero run, one using the
fixed_beta_qpso run for that same scenario. Show both vehicles moving at once
(split map or two small maps side by side), with each run's own completion
time counting up live. When one finishes, freeze it and highlight the time
difference ("Adaptive finished 2m 14s faster in chaotic traffic"). This is
the single visual that makes your research claim legible in ten seconds
without reading a chart.
```

## Prompt 7 — Onboarding overlay (this is what makes it usable by "anyone")

```
Add a dismissible intro overlay shown on first load: 3-4 short sentences in
plain language — what this system does (delivery vehicle, multiple stops,
changing traffic), what to watch for (the unpredictability number and the
event feed), and a "Got it — show me" button that dismisses it and starts
playback. Also add a persistent small "?" button that re-opens this overlay,
in case a judge wants to re-read it mid-demo.
```

## Prompt 8 — Playback controls

```
Add standard playback controls: play/pause, a speed selector (1x/2x/4x — 4x
useful if a judge wants to see a full run quickly), a scrub bar showing
progress through the run, and a restart button. Make sure switching speed or
scrubbing doesn't desync the map, metrics panel, and event feed from each
other — they all key off the same single "current playback time" value.
```

## Prompt 9 — Reliability pass (do this last, don't skip it)

```
Review index.html and server.py for anything that assumes a live SUMO/TraCI
connection or a specific machine's file paths. The demo must run entirely
from the bundled hero-run JSON files with zero live dependencies. Add a
clear fallback: if a fetch to /api/runs fails for any reason, show a friendly
error state rather than a blank page or console error. Test the whole flow
with the backend freshly started and a cold browser cache — this is exactly
the condition it'll be run under at judging time.
```

---

## If you're short on time, cut in this order

1. Drop the 4x speed option and split-screen comparison — keep just a toggle
   between the two runs instead of simultaneous playback.
2. Drop the info-icon tooltips — keep the plain-language labels as the primary
   text (don't drop those; that's the part that makes it usable by anyone).
3. Drop live-mode entirely if you were planning one — the replay demo is not
   a fallback, it should be treated as the primary and only demo path.
4. **Never cut:** Prompt 9's reliability pass. A beautiful frontend that fails
   to load on the judging laptop scores worse than a plain one that works.
