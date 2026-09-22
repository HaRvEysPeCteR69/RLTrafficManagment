# Bundled demo runs

Place the exported hero-run JSON files here. `server.py` serves each file as
`/api/runs/<filename-without-.json>` and the browser can also fall back to
`frontend_data/<filename>.json` when using a static web server.

Generate matched hero data with:

```sh
python3 export_for_frontend.py bundle --manifest frontend_runs.json \
  --net networks/delhi/delhi_intersection.net.xml --output-dir frontend_data
```
