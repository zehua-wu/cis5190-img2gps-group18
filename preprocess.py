# submission/preprocess.py
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Iterable, List, Tuple


def _resolve_column(header: List[str], aliases: List[str]) -> int:
    lower = [h.strip().lower() for h in header]
    for a in aliases:
        a = a.lower()
        if a in lower:
            return lower.index(a)
    raise KeyError(f"Could not find any of columns {aliases} in CSV header: {header}")


def _candidate_roots(csv_path: Path) -> List[Path]:
    
    roots = [
        csv_path.parent,
        csv_path.parent / "images",
        csv_path.parent / "imgs",
        csv_path.parent / "img",
        csv_path.parent.parent / "images",
        csv_path.parent.parent / "imgs",
        csv_path.parent.parent / "img",
        csv_path.parent / "reference" / "images",
    ]
    
    seen = set()
    out = []
    for r in roots:
        rp = r.resolve()
        if rp not in seen:
            seen.add(rp)
            out.append(rp)
    return out


def _resolve_image_path(name: str, roots: List[Path]) -> str:
    p = Path(name)
    if p.is_absolute() and p.exists():
        return str(p)

    for r in roots:
        cand = (r / name)
        if cand.exists():
            return str(cand)

    base = Path(name).name
    for r in roots:
        cand = (r / base)
        if cand.exists():
            return str(cand)

    return str((roots[0] / Path(name).name) if roots else Path(name))


def prepare_data(path: str) -> Tuple[List[Any], List[Any]]:
    """
    Leaderboard requirement:
      return (X, y)
        X: list of model-ready inputs -> 我这里用 "图片路径字符串"
        y: list of [lat, lon]（评测会自己从csv读label；但我们仍返回，便于本地debug）
    """
    csv_path = Path(path)
    roots = _candidate_roots(csv_path)

    X: List[Any] = []
    y: List[Any] = []

    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)

        
        try:
            idx_file = _resolve_column(header, ["file_name", "filename", "image", "img", "path"])
            idx_lat = _resolve_column(header, ["latitude", "lat"])
            idx_lon = _resolve_column(header, ["longitude", "lon", "lng"])
        except Exception:
            idx_file, idx_lat, idx_lon = 0, 1, 2

        for row in reader:
            if not row:
                continue
            
            if len(row) <= max(idx_file, idx_lat, idx_lon):
                continue

            file_name = row[idx_file].strip()
            lat = float(row[idx_lat])
            lon = float(row[idx_lon])

            img_path = _resolve_image_path(file_name, roots)
            X.append(img_path)
            y.append([lat, lon])

    return X, y
