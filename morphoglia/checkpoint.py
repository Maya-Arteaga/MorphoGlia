from __future__ import annotations

# MG_RESUME_CHECKPOINT_V1

from dataclasses import asdict, is_dataclass
from pathlib import Path
import hashlib
import json


def _normal(value):
    if is_dataclass(value):
        return _normal(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _normal(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, (list, tuple)):
        return [_normal(v) for v in value]
    if isinstance(value, set):
        return sorted((_normal(v) for v in value), key=repr)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _normal(item())
        except Exception:
            pass
    return repr(value)


def fingerprint(*parts) -> str:
    payload = json.dumps(_normal(parts), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def source_digest(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_identity(path: str | Path) -> dict:
    path = Path(path)
    stat = path.stat()
    return {
        "path": str(path.expanduser().resolve()),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


class CheckpointJournal:
    def __init__(self, *, output_dir, stage, resume, stage_signature):
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.stage = str(stage)
        self.resume = bool(resume)
        self.stage_signature = str(stage_signature)
        root = self.output_dir / "Technical_Record" / "Checkpoints"
        root.mkdir(parents=True, exist_ok=True)
        self.path = root / f"{self.stage}.jsonl"
        self.records = {}
        if self.resume and self.path.is_file():
            for line in self.path.read_text(encoding="utf-8", errors="replace").splitlines():
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                if (
                    row.get("status") == "complete"
                    and str(row.get("stage_signature", "")) == self.stage_signature
                    and str(row.get("item_id", ""))
                ):
                    self.records[str(row["item_id"])] = row
        self._compact()

    def _token(self, path):
        path = Path(path).expanduser().resolve()
        try:
            return str(path.relative_to(self.output_dir))
        except ValueError:
            return str(path)

    def _compact(self):
        tmp = self.path.with_name(self.path.name + ".tmp")
        try:
            with tmp.open("w", encoding="utf-8", newline="\n") as fh:
                for item_id in sorted(self.records):
                    fh.write(json.dumps(self.records[item_id], sort_keys=True, separators=(",", ":")) + "\n")
            tmp.replace(self.path)
        finally:
            if tmp.exists():
                tmp.unlink()

    def reusable_record(self, *, item_id, item_signature, outputs):
        if not self.resume:
            return None
        row = self.records.get(str(item_id))
        if row is None or str(row.get("item_signature", "")) != str(item_signature):
            return None
        outputs = [Path(p) for p in outputs]
        saved = row.get("outputs", [])
        if len(outputs) != len(saved):
            return None
        for path, old in zip(outputs, saved):
            try:
                stat = path.stat()
            except OSError:
                return None
            if not path.is_file() or stat.st_size <= 0:
                return None
            if self._token(path) != str(old.get("path", "")):
                return None
            if int(stat.st_size) != int(old.get("size", -1)):
                return None
            if int(stat.st_mtime_ns) != int(old.get("mtime_ns", -1)):
                return None
        return dict(row)

    def commit(self, *, item_id, item_signature, outputs, metadata=None):
        saved = []
        for path in [Path(p) for p in outputs]:
            stat = path.stat()
            if not path.is_file() or stat.st_size <= 0:
                raise RuntimeError(f"Cannot checkpoint missing or empty output: {path}")
            saved.append({
                "path": self._token(path),
                "size": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
            })
        row = {
            "stage": self.stage,
            "stage_signature": self.stage_signature,
            "item_id": str(item_id),
            "item_signature": str(item_signature),
            "status": "complete",
            "outputs": saved,
            "metadata": _normal(metadata or {}),
        }
        self.records[str(item_id)] = row
        with self.path.open("a", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
            fh.flush()


def atomic_write_bytes(path: str | Path, data: bytes) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.stem}.tmp{path.suffix}")
    if tmp.exists():
        tmp.unlink()
    try:
        tmp.write_bytes(data)
        tmp.replace(path)
    finally:
        if tmp.exists():
            tmp.unlink()
    return path
