import argparse, json, time
from pathlib import Path
import yaml

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--outdir", required=True)
    args = p.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    result = {"ok": True, "cfg": cfg, "ts": time.time()}
    Path(args.outdir, "metrics.json").write_text(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
