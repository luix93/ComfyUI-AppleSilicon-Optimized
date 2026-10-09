import json
from pathlib import Path

root = Path(__file__).resolve().parent
for path in sorted(root.glob('*.json')):
    data=json.loads(path.read_text())
    if 'state' not in data:continue
    print(path.stem, {key: data['state'].get(key) for key in
          ('completed','stopped','stage','max_driver_gib','max_live_gib','max_rss_gib')})
    for record in data['records']:
        print(' ', record['repeat'], round(record['total_seconds'],3),
              {key:round(value,3) for key,value in record['timings'].items()},
              {key:record[key][:12] for key in
               ('encoded_sha256','sampled_sha256','frames_sha256') if key in record})
