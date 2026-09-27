"""
Engine service layer.

Sits between the HTTP API and the engine, and owns the things a web
request cannot: a catalog that survives across requests, analysis caching,
and long-running jobs that outlive the connection that started them.

Two decisions worth stating, because both are load-bearing:

**Analysis is cached by content hash, not by filename.** Beat analysis is
the expensive part of the pipeline -- separation, beat tracking, chord
estimation and groove extraction on a full instrumental -- and it is also
completely deterministic for a given file and analyser version. Re-running
it per request would make the product feel slow for no reason. Keying on
the audio's hash means renaming or re-uploading the same file costs
nothing, and bumping the analyser version invalidates exactly the entries
that need it.

**Renders run as jobs, not as requests.** A render takes tens of seconds
to minutes. Holding an HTTP connection open for that is fragile and gives
the caller nothing to show meanwhile, so work happens on a worker thread
and progress is polled. The job record carries stage-level progress so the
interface can say what the engine is actually doing rather than showing an
indeterminate spinner.
"""

from __future__ import annotations

import logging
import os
import threading
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional

log = logging.getLogger("mixengine.service")

# The salt for every cache key. Taken from the DNA schema version rather
# than set independently: the CLI keys its output on that, and two salts
# meant the same beat imported through the CLI and through the web
# interface produced two catalog entries.
from ..config import DNA_SCHEMA_VERSION as ANALYSIS_VERSION

JOB_QUEUED = "queued"
JOB_RUNNING = "running"
JOB_DONE = "done"
JOB_FAILED = "failed"


def file_hash(path: str, chunk: int = 1 << 20) -> str:
    """Content hash of a file. Streamed, so large audio does not blow memory."""
    from ..core import audio_io
    return audio_io.content_key(path, ANALYSIS_VERSION, chunk)


@dataclass
class Job:
    id: str
    kind: str
    status: str = JOB_QUEUED
    stage: str = ""
    progress: float = 0.0
    message: str = ""
    result: Optional[dict] = None
    error: str = ""
    created_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["elapsed_s"] = round((self.finished_at or time.time()) - self.created_at, 1)
        return d


class Workspace:
    """On-disk layout for one installation."""

    def __init__(self, root: str = "./data"):
        self.root = os.path.abspath(root)
        for sub in ("beats", "vocals", "dna/beats", "dna/vocals", "stems",
                    "outputs", "cache", "uploads"):
            os.makedirs(os.path.join(self.root, sub), exist_ok=True)

    def path(self, *parts: str) -> str:
        return os.path.join(self.root, *parts)

    def beat_dna_path(self, key: str) -> str:
        return self.path("dna", "beats", "%s.json" % key)

    def vocal_dna_path(self, key: str) -> str:
        return self.path("dna", "vocals", "%s.json" % key)


class EngineService:
    """Everything the API needs, with the engine imported lazily.

    The heavy imports -- librosa, soundfile, the analysis stack -- are
    deferred until first use so the server starts instantly and a machine
    missing an optional dependency still serves the catalog and reports
    what it is missing, rather than failing at import time with a traceback.
    """

    def __init__(self, root: str = "./data"):
        self.ws = Workspace(root)
        self._jobs: Dict[str, Job] = {}
        self._lock = threading.Lock()
        self._catalog_cache: Optional[List[dict]] = None

    # -- capabilities ------------------------------------------------------

    def capabilities(self) -> dict:
        from ..core.capabilities import CAPS
        d = CAPS.to_dict()
        d["tier"] = CAPS.tier
        d["can_render"] = CAPS.can_render
        d["can_separate"] = CAPS.can_separate
        d["can_stretch_well"] = CAPS.can_stretch_well
        missing = []
        if not CAPS.can_stretch_well:
            missing.append({"name": "rubberband",
                            "cost": "formant-preserving stretch and pitch shift"})
        if not CAPS.madmom:
            missing.append({"name": "madmom",
                            "cost": "downbeat tracking for bar-accurate placement"})
        if not CAPS.can_separate:
            missing.append({"name": "demucs",
                            "cost": "stems, so drum-safe pitch shift and ducking"})
        if not CAPS.torchcrepe:
            missing.append({"name": "torchcrepe", "cost": "accurate pitch tracking"})
        if not CAPS.pyloudnorm:
            missing.append({"name": "pyloudnorm", "cost": "true LUFS metering"})
        d["missing"] = missing
        return d

    # -- jobs --------------------------------------------------------------

    def _new_job(self, kind: str) -> Job:
        job = Job(id=uuid.uuid4().hex[:12], kind=kind)
        with self._lock:
            self._jobs[job.id] = job
        return job

    def job(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def jobs(self) -> List[dict]:
        with self._lock:
            return [j.to_dict() for j in
                    sorted(self._jobs.values(), key=lambda x: -x.created_at)]

    def _run_async(self, job: Job, fn: Callable[[Job], dict]) -> Job:
        def runner():
            job.status = JOB_RUNNING
            try:
                job.result = fn(job)
                job.status = JOB_DONE
                job.progress = 1.0
                job.stage = "complete"
            except Exception as e:
                job.status = JOB_FAILED
                job.error = "%s: %s" % (type(e).__name__, e)
                # Keep the traceback server-side; the client gets the summary.
                log.exception("job %s (%s) failed", job.id, job.kind)
            finally:
                job.finished_at = time.time()
        threading.Thread(target=runner, daemon=True, name="job-%s" % job.id).start()
        return job

    # -- beats -------------------------------------------------------------

    def analyze_beat(self, audio_path: str, *, metadata: Optional[dict] = None,
                     force: bool = False, separate: bool = False,
                     job: Optional[Job] = None) -> dict:
        """Analyse one beat, reusing a cached result when the audio matches."""
        from ..analysis import beat_dna
        from ..core import audio_io

        key = file_hash(audio_path)
        cache = self.ws.beat_dna_path(key)
        if not force and os.path.exists(cache):
            cached = audio_io.read_json(cache)
            if cached is not None and beat_dna.is_current(cached):
                why = beat_dna.can_improve(cached, want_stems=separate)
                if why is None:
                    cached["cached"] = True
                    return cached
                log.info("re-analysing %s: %s", os.path.basename(audio_path), why)

        if job:
            job.stage, job.progress = "analysing beat", 0.15
        meta = dict(metadata or {})
        beat_id = meta.get("beat_id") or "beat-%s" % key
        stems_dir = self.ws.path("stems") if separate else None

        dna = beat_dna.extract(audio_path, metadata=meta, stems_dir=stems_dir,
                               do_separation=separate, beat_id=beat_id)
        dna["cache_key"] = key
        dna["cached"] = False
        audio_io.write_json(cache, dna)
        self._catalog_cache = None
        return dna

    def catalog(self, refresh: bool = False) -> List[dict]:
        from ..core import audio_io
        from ..analysis import beat_dna
        if self._catalog_cache is not None and not refresh:
            return self._catalog_cache
        out: List[dict] = []
        d = self.ws.path("dna", "beats")
        for f in sorted(os.listdir(d)) if os.path.isdir(d) else []:
            if not f.endswith(".json"):
                continue
            doc = audio_io.read_json(os.path.join(d, f))
            if doc is not None and beat_dna.is_current(doc):
                out.append(doc)
        self._catalog_cache = out
        return out

    def catalog_summary(self) -> List[dict]:
        """Compact rows for the catalog table -- no beat grids or curves."""
        rows = []
        for b in self.catalog():
            key = b.get("key") or {}
            rows.append({
                "beat_id": b.get("beat_id"),
                "title": b.get("title") or b.get("beat_id"),
                "genre": b.get("genre"),
                "bpm": b.get("bpm"),
                "bpm_source": b.get("bpm_source"),
                "key_name": key.get("name"),
                "camelot": b.get("camelot"),
                "is_atonal": b.get("is_atonal"),
                "pocket_score": b.get("pocket_score"),
                "grid_stability": b.get("grid_stability"),
                "swing_ratio": b.get("swing_ratio"),
                "duration_s": b.get("duration_s"),
                "has_stems": b.get("has_stems"),
                "n_sections": len(b.get("sections") or []),
            })
        return rows

    # -- vocals ------------------------------------------------------------

    def analyze_vocal(self, audio_path: str, *, user_bpm: Optional[float] = None,
                      user_key: Optional[str] = None, force: bool = False,
                      recorded_over: Optional[str] = None,
                      job: Optional[Job] = None) -> dict:
        """Analyse a take.

        `recorded_over` names the catalog beat that was playing while the
        take was recorded. It feeds two stages that cannot work without it:
        bleed cancellation, which needs the reference signal to subtract,
        and tempo, which becomes the beat's tempo rather than an estimate.
        """
        from ..analysis import vocal_dna
        from ..core import audio_io

        ref = None
        if recorded_over:
            ref = next((b for b in self.catalog()
                        if b.get("beat_id") == recorded_over), None)

        key = file_hash(audio_path)
        if user_bpm or user_key or ref:
            key = "%s-%s-%s-%s" % (key, user_bpm or "", user_key or "",
                                   (ref or {}).get("cache_key") or recorded_over or "")
        cache = self.ws.vocal_dna_path(key)
        if not force and os.path.exists(cache):
            cached = audio_io.read_json(cache)
            if cached and cached.get("status") == "ok":
                why = vocal_dna.can_improve(cached)
                if why is None:
                    cached["cached"] = True
                    return cached
                log.info("re-analysing %s: %s", os.path.basename(audio_path), why)

        if job:
            job.stage, job.progress = "analysing vocal", 0.2
        # The restored take is what gets rendered. Without a path for it
        # the analysis was made on the cleaned vocal and the mix on the
        # noisy original, which is how a take's fan noise reached the
        # master untouched and then boosted.
        dna = vocal_dna.extract(
            audio_path, user_bpm=user_bpm, user_key=user_key, do_separation=True,
            reference_beat_path=(ref or {}).get("source_path"),
            reference_beat_dna=ref,
            conditioned_out=self.ws.path("vocals", "%s-conditioned.wav" % key))
        if recorded_over and ref is None:
            dna.setdefault("warnings", []).append(
                f"recorded_over={recorded_over!r} is not in the catalog; "
                f"analysed without a reference beat")
        dna["cache_key"] = key
        dna["cached"] = False
        audio_io.write_json(cache, dna)
        return dna

    # -- matching ----------------------------------------------------------

    def match(self, vocal_dna_doc: dict, n: int = 5) -> dict:
        from ..analysis import matching
        cat = self.catalog()
        if not cat:
            return {"matches": [], "catalog_size": 0,
                    "message": "No analysed beats yet. Add beats to the catalog first."}
        report = matching.find_matches(vocal_dna_doc, cat, n=n)
        return report.to_dict()

    # -- render ------------------------------------------------------------

    def start_render(self, vocal_path: str, beat_ids: List[str],
                     *, variants: int = 1, user_bpm: Optional[float] = None,
                     user_key: Optional[str] = None) -> Job:
        job = self._new_job("render")

        def work(j: Job) -> dict:
            from ..audio import pipeline
            j.stage, j.progress, j.message = "analysing vocal", 0.1, \
                "Measuring pitch, phrasing and timing"
            vdna = self.analyze_vocal(vocal_path, user_bpm=user_bpm,
                                      user_key=user_key, job=j)
            if vdna.get("status") != "ok":
                raise RuntimeError(vdna.get("error") or "vocal analysis failed")

            cat = self.catalog()
            chosen = [b for b in cat if b.get("beat_id") in set(beat_ids)] \
                if beat_ids else cat
            if not chosen:
                raise RuntimeError("none of the requested beats are in the catalog")

            j.stage, j.progress, j.message = "rendering", 0.35, \
                "Aligning, tuning, mixing and mastering"
            out_dir = self.ws.path("outputs", j.id)
            result = pipeline.run(vocal_path, chosen, out_dir,
                                  n_beats=max(1, len(chosen)),
                                  variants_per_beat=variants, vdna=vdna)
            j.stage, j.progress = "finished", 0.95

            for r in result.get("renders", []):
                p = r.get("path") or ""
                if p:
                    r["download"] = "/api/audio/%s/%s" % (j.id, os.path.basename(p))
            result["vocal_summary"] = vdna.get("summary")
            return result

        return self._run_async(job, work)

    def start_session_render(self, vocal_path: str, beat_path: str,
                             *, intents: Optional[Any] = None) -> Job:
        """One vocal, one beat, one run.

        The dashboard collects both files and whatever the user wants to
        say about them, then calls this once. Analysis of each input
        happens inside the job rather than on upload: a person choosing a
        file has not asked for anything to be computed yet, and firing a
        separate analysis per upload meant the engine ran three times to
        make one song.
        """
        job = self._new_job("render")

        def work(j: Job) -> dict:
            from ..audio import pipeline

            j.stage, j.progress, j.message = "intake", 0.05, \
                "Listening to the beat"
            bdna = self.analyze_beat(beat_path, job=j)
            if bdna.get("status") != "ok":
                raise RuntimeError(bdna.get("error") or "beat analysis failed")

            j.stage, j.progress, j.message = "intake", 0.2, \
                "Listening to the vocal"
            user_bpm = getattr(intents, "bpm", None) if intents else None
            user_key = getattr(intents, "key", None) if intents else None
            vdna = self.analyze_vocal(vocal_path, user_bpm=user_bpm,
                                      user_key=user_key, job=j)
            if vdna.get("status") != "ok":
                raise RuntimeError(vdna.get("error") or "vocal analysis failed")

            j.stage, j.progress, j.message = "transform", 0.4, \
                "Placing, mixing and mastering"
            out_dir = self.ws.path("outputs", j.id)
            result = pipeline.run(vocal_path, [bdna], out_dir, n_beats=1,
                                  variants_per_beat=1, vdna=vdna,
                                  intents=intents)

            j.stage, j.progress = "check", 0.95
            for r in result.get("renders", []):
                p = r.get("path") or ""
                if p:
                    r["download"] = "/api/audio/%s/%s" % (j.id,
                                                          os.path.basename(p))
            result["vocal_summary"] = vdna.get("summary")
            return result

        return self._run_async(job, work)

    def start_beat_import(self, paths: List[str],
                          metadata_by_file: Optional[Dict[str, dict]] = None,
                          separate: bool = False) -> Job:
        job = self._new_job("import")

        def work(j: Job) -> dict:
            done, failed = [], []
            for i, p in enumerate(paths):
                j.stage = "analysing %s" % os.path.basename(p)
                j.progress = (i + 0.5) / max(len(paths), 1)
                try:
                    meta = (metadata_by_file or {}).get(os.path.basename(p), {})
                    dna = self.analyze_beat(p, metadata=meta, separate=separate)
                    done.append({"beat_id": dna.get("beat_id"),
                                 "title": dna.get("title"),
                                 "bpm": dna.get("bpm"),
                                 "cached": dna.get("cached", False)})
                except Exception as e:
                    failed.append({"file": os.path.basename(p), "error": str(e)})
            self._catalog_cache = None
            return {"imported": done, "failed": failed,
                    "catalog_size": len(self.catalog(refresh=True))}

        return self._run_async(job, work)

    # -- takes / coaching --------------------------------------------------

    def preflight(self, audio_path: str, *,
                  beat_id: Optional[str] = None,
                  vocal_dna_doc: Optional[dict] = None) -> dict:
        """Qualify a room and gain setting from a short silent recording.

        When a beat is named, the guidance becomes musical as well as
        technical: key, tempo, where the bars fall, and -- the most valuable
        item in the product -- whether that key puts the melody outside the
        singer's measured range. No downstream processing fixes a phrase
        sung above someone's tessitura; transposing the beat before the
        take does.
        """
        from ..core import audio_io
        from ..capture.coach import preroll_guidance
        from ..core.keys import Key

        y, sr, q = audio_io.load(audio_path, sr=48000)

        bdna = None
        if beat_id:
            bdna = next((b for b in self.catalog()
                         if b.get("beat_id") == beat_id), None)

        scale: List[float] = []
        key_name = None
        if bdna and (bdna.get("key") or {}).get("pc") is not None:
            k = bdna["key"]
            key_name = k.get("name")
            key = Key(int(k["pc"]), str(k.get("mode", "minor")))
            scale = [60 + p for p in key.scale_pcs]

        voice = (vocal_dna_doc or {}).get("voice") or {}
        cues = preroll_guidance(
            beat_key_name=key_name,
            beat_bpm=float((bdna or {}).get("bpm") or 0.0),
            beat_bars=int((bdna or {}).get("duration_bars") or 0),
            scale_midi=scale,
            voice_low_midi=float(voice.get("tessitura_low_midi") or 0.0),
            voice_high_midi=float(voice.get("tessitura_high_midi") or 0.0),
            performance_type=str((vocal_dna_doc or {}).get(
                "performance_type") or ""),
            rt60_s=float(q.estimated_rt60_s),
            noise_floor_db=float(q.noise_floor_db),
            has_beat=bool(bdna))
        return {"quality": q.to_dict(),
                "beat": _beat_brief(bdna) if bdna else None,
                "cues": [c.to_dict() for c in cues],
                "ok": not any(c.severity == "fatal" for c in cues)}

    def review_take(self, audio_path: str,
                    beat_dna_doc: Optional[dict] = None) -> dict:
        """Post-take report: what the recording was like, and whether to keep it."""
        from ..core import audio_io
        from ..capture.realtime import FrameAnalyzer
        from ..capture.coach import take_report
        from ..core.keys import Key

        y, sr, _ = audio_io.load(audio_path, sr=48000)
        mono = y.mean(axis=1) if getattr(y, "ndim", 1) > 1 else y
        fa = FrameAnalyzer(sr=48000)
        frames = []
        step = 48000
        for i in range(0, len(mono), step):
            frames.extend(fa.push(mono[i:i + step]))

        scale = []
        if beat_dna_doc and (beat_dna_doc.get("key") or {}).get("pc") is not None:
            k = beat_dna_doc["key"]
            key = Key(int(k["pc"]), str(k.get("mode", "minor")))
            scale = [60 + p for p in key.scale_pcs]
        rep = take_report(frames, scale_midi=scale,
                          noise_floor_db=fa.noise_floor_db)
        return rep.to_dict()


def _beat_brief(bdna: dict) -> dict:
    """The few fields a recording session actually needs from a beat."""
    return {
        "beat_id": bdna.get("beat_id"),
        "title": bdna.get("title") or bdna.get("beat_id"),
        "source_name": os.path.basename(bdna.get("source_path") or ""),
        "bpm": bdna.get("bpm"),
        "key": (bdna.get("key") or {}).get("name"),
        "camelot": bdna.get("camelot"),
        "bars": bdna.get("duration_bars"),
        "beats_per_bar": bdna.get("beats_per_bar", 4),
        "duration_s": bdna.get("duration_s"),
        "genre": bdna.get("genre"),
        "downbeats": (bdna.get("downbeats") or [])[:64],
        "sections": bdna.get("sections") or [],
    }
