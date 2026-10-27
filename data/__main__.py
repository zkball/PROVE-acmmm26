from . import status

if __name__ == "__main__":
    for name, info in status().items():
        print(f"{name} ({info['name']}) -> {info['dir']}")
        for split, s in info["splits"].items():
            flag = "ready" if s["prepared"] else ("jsonl only" if s["exists"] else "missing")
            n = s["n"] if s["n"] is not None else "-"
            print(f"  {split:5s}  {flag:12s}  n={n}")
