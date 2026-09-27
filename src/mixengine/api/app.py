"""
HTTP API.

Thin by design: every endpoint validates its inputs, delegates to
`EngineService`, and returns JSON. No engine logic lives here, so the API
can be replaced or supplemented (CLI, queue worker, gRPC) without touching
anything that makes musical decisions.

Uploads are written to disk before analysis rather than held in memory,
because the engine works on files -- separation shells out to a subprocess,
Rubber Band takes paths -- and because a 10-minute stereo upload is large
enough that buffering several at once is a real risk.
"""

from __future__ import annotations

import hashlib
import logging
import os
import tempfile
from typing import List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware

from . import service
from .service import EngineService

log = logging.getLogger("mixengine.api")

AUDIO_EXTS = {".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg", ".aiff", ".aif"}
MAX_UPLOAD_BYTES = 512 * 1024 * 1024


def create_app(data_root: str = "./data") -> FastAPI:
    app = FastAPI(title="mixengine", version="2.0.0",
                  description="Vocal + beat to studio-ready song.")
    svc = EngineService(data_root)

    # Local-first tool: the interface is served from the same origin, and
    # anything else is a deliberate development setup.
    app.add_middleware(CORSMiddleware, allow_origins=["*"],
                       allow_methods=["*"], allow_headers=["*"])
    app.add_middleware(IsolationMiddleware)

    # ── helpers ───────────────────────────────────────────────────────────

    def _save_upload(f: UploadFile, subdir: str) -> str:
        ext = os.path.splitext(f.filename or "")[1].lower()
        if ext not in AUDIO_EXTS:
            raise HTTPException(400, "unsupported audio format: %r" % ext)
        safe = os.path.basename(f.filename or "upload")
        # Stream to a temporary file while hashing, then place the result
        # under a directory named by its content. Saving straight to
        # `<subdir>/<name>` meant an upload called beat.wav replaced the
        # library's beat.wav; keying the directory on content means the
        # same file uploaded twice lands in one place and two different
        # files never collide, while the filename -- which becomes the
        # vocal or beat id -- is kept as the user gave it.
        root = svc.ws.path(subdir)
        os.makedirs(root, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=root, prefix=".upload-", suffix=ext)
        digest = hashlib.sha256()
        size = 0
        try:
            with os.fdopen(fd, "wb") as out:
                while True:
                    chunk = f.file.read(1 << 20)
                    if not chunk:
                        break
                    size += len(chunk)
                    if size > MAX_UPLOAD_BYTES:
                        raise HTTPException(413, "file exceeds the 512 MB limit")
                    digest.update(chunk)
                    out.write(chunk)
            if size == 0:
                raise HTTPException(400, "uploaded file is empty")
            dest = svc.ws.path(subdir, digest.hexdigest()[:12], safe)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            os.replace(tmp, dest)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)
        return dest

    # ── meta ──────────────────────────────────────────────────────────────

    @app.get("/api/health")
    def health():
        return {"ok": True, "version": app.version}

    @app.get("/api/capabilities")
    def capabilities():
        return svc.capabilities()

    # ── catalog ───────────────────────────────────────────────────────────

    @app.get("/api/catalog")
    def catalog():
        return {"beats": svc.catalog_summary()}

    @app.get("/api/catalog/{beat_id}")
    def catalog_detail(beat_id: str):
        for b in svc.catalog():
            if b.get("beat_id") == beat_id:
                # Beat grids and reference curves are large and the interface
                # never renders them; sections and chords it does.
                return {k: v for k, v in b.items()
                        if k not in ("beats", "downbeats", "reference_curve")}
        raise HTTPException(404, "no such beat")

    @app.post("/api/beats")
    def add_beats(files: List[UploadFile] = File(...),
                  genre: Optional[str] = Form(None),
                  separate: bool = Form(False)):
        paths, meta = [], {}
        for f in files:
            p = _save_upload(f, "beats")
            paths.append(p)
            if genre:
                meta[os.path.basename(p)] = {"genre": genre,
                                             "title": os.path.splitext(
                                                 os.path.basename(p))[0]}
        job = svc.start_beat_import(paths, meta, separate=separate)
        return job.to_dict()

    # ── vocal ─────────────────────────────────────────────────────────────

    @app.post("/api/vocal/analyze")
    def analyze_vocal(file: UploadFile = File(...),
                      bpm: Optional[float] = Form(None),
                      key: Optional[str] = Form(None),
                      recorded_over: Optional[str] = Form(None)):
        path = _save_upload(file, "vocals")
        dna = svc.analyze_vocal(path, user_bpm=bpm, user_key=key,
                                recorded_over=recorded_over)
        if dna.get("status") != "ok":
            raise HTTPException(422, dna.get("error") or "vocal analysis failed")
        # Dense per-note and per-onset arrays are not used by the interface.
        slim = {k: v for k, v in dna.items()
                if k not in ("notes", "onsets_s", "phrases")}
        slim["path"] = path
        return slim

    @app.post("/api/vocal/match")
    def match_vocal(file: UploadFile = File(...),
                    bpm: Optional[float] = Form(None),
                    key: Optional[str] = Form(None),
                    n: int = Form(5),
                    recorded_over: Optional[str] = Form(None)):
        path = _save_upload(file, "vocals")
        dna = svc.analyze_vocal(path, user_bpm=bpm, user_key=key,
                                recorded_over=recorded_over)
        if dna.get("status") != "ok":
            raise HTTPException(422, dna.get("error") or "vocal analysis failed")
        return {"vocal": {k: v for k, v in dna.items()
                          if k not in ("notes", "onsets_s", "phrases")},
                "path": path,
                "match": svc.match(dna, n=int(n))}

    # ── take review ───────────────────────────────────────────────────────

    @app.post("/api/take/preflight")
    def preflight(file: UploadFile = File(...),
                  beat_id: Optional[str] = Form(None)):
        return svc.preflight(_save_upload(file, "uploads"), beat_id=beat_id)

    @app.get("/api/take/guidance")
    def guidance(beat_id: Optional[str] = None):
        """Pre-roll guidance without a room recording.

        Split out from `preflight` because the two answer different
        questions and the client needs them at different moments: this one
        is what to know about the beat, and can be shown as soon as a beat
        is picked. The room check needs the singer to record silence first,
        which is a step they may skip.
        """
        from ..capture.coach import preroll_guidance
        from ..core.keys import Key

        bdna = None
        if beat_id:
            bdna = next((b for b in svc.catalog()
                         if b.get("beat_id") == beat_id), None)
        if beat_id and bdna is None:
            raise HTTPException(404, "no such beat")

        scale, key_name = [], None
        if bdna and (bdna.get("key") or {}).get("pc") is not None:
            k = bdna["key"]
            key_name = k.get("name")
            scale = [60 + p for p in Key(int(k["pc"]),
                                         str(k.get("mode", "minor"))).scale_pcs]
        cues = preroll_guidance(
            beat_key_name=key_name,
            beat_bpm=float((bdna or {}).get("bpm") or 0.0),
            beat_bars=int((bdna or {}).get("duration_bars") or 0),
            scale_midi=scale, has_beat=bool(bdna))
        return {"beat": service._beat_brief(bdna) if bdna else None,
                "cues": [c.to_dict() for c in cues]}

    @app.post("/api/take/review")
    def review(file: UploadFile = File(...),
               beat_id: Optional[str] = Form(None)):
        path = _save_upload(file, "vocals")
        bdna = None
        if beat_id:
            bdna = next((b for b in svc.catalog()
                         if b.get("beat_id") == beat_id), None)
        return {"path": path, **svc.review_take(path, bdna)}

    # ── render ────────────────────────────────────────────────────────────

    @app.post("/api/render")
    def render(vocal: Optional[UploadFile] = File(None),
               beat: Optional[UploadFile] = File(None),
               vocal_path: Optional[str] = Form(None),
               beat_ids: str = Form(""),
               variants: int = Form(1),
               bpm: Optional[float] = Form(None),
               key: Optional[str] = Form(None),
               vocal_state: Optional[str] = Form(None),
               relationship: Optional[str] = Form(None),
               tune: Optional[str] = Form(None),
               timing: Optional[str] = Form(None),
               space: Optional[str] = Form(None),
               separate: Optional[str] = Form(None),
               loudness: Optional[str] = Form(None),
               nudge: Optional[str] = Form(None)):
        """Render a song.

        Two shapes, because two callers need different things. The
        dashboard posts both files and whatever the user said about them,
        and gets one run. The CLI and older clients post a `vocal_path`
        already on disk plus catalog beat ids.
        """
        from ..core.intents import Intents
        try:
            intents = Intents.from_dict({
                "vocal_state": vocal_state, "relationship": relationship,
                "tune": tune, "timing": timing, "space": space,
                "separate": separate, "loudness": loudness, "nudge": nudge,
                "bpm": bpm, "key": key})
        except ValueError as e:
            raise HTTPException(422, str(e))

        if vocal is not None and beat is not None:
            job = svc.start_session_render(_save_upload(vocal, "vocals"),
                                           _save_upload(beat, "beats"),
                                           intents=intents)
            return job.to_dict()

        if not vocal_path or not os.path.exists(vocal_path):
            raise HTTPException(400, "send a vocal and a beat, or a "
                                     "vocal_path that exists")
        ids = [b for b in beat_ids.split(",") if b.strip()]
        job = svc.start_render(vocal_path, ids, variants=int(variants),
                               user_bpm=bpm, user_key=key)
        return job.to_dict()

    @app.get("/api/jobs")
    def jobs():
        return {"jobs": svc.jobs()}

    @app.get("/api/jobs/{job_id}")
    def job_status(job_id: str):
        j = svc.job(job_id)
        if j is None:
            raise HTTPException(404, "no such job")
        return j.to_dict()

    @app.get("/api/audio/{job_id}/{name}")
    def audio(job_id: str, name: str):
        # Resolve and confine: a job id and filename come from the client, so
        # the joined path must be proven to sit inside the outputs directory
        # before anything is served from it.
        base = os.path.realpath(svc.ws.path("outputs"))
        target = os.path.realpath(os.path.join(base, job_id, os.path.basename(name)))
        if not target.startswith(base + os.sep) or not os.path.exists(target):
            raise HTTPException(404, "no such file")
        return FileResponse(target, media_type="audio/wav", filename=name)

    @app.get("/api/source/{kind}/{name}")
    def source_audio(kind: str, name: str):
        if kind not in ("beats", "vocals"):
            raise HTTPException(404, "no such source")
        base = os.path.realpath(svc.ws.path(kind))
        target = os.path.realpath(os.path.join(base, os.path.basename(name)))
        if not target.startswith(base + os.sep) or not os.path.exists(target):
            raise HTTPException(404, "no such file")
        return FileResponse(target)

    # ── interface ─────────────────────────────────────────────────────────

    web_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "web")
    if os.path.isdir(web_dir):
        app.mount("/static", StaticFiles(directory=web_dir), name="static")

        @app.get("/", response_class=HTMLResponse)
        def index():
            with open(os.path.join(web_dir, "index.html")) as f:
                return HTMLResponse(f.read(), headers=_ISOLATION_HEADERS)

    app.state.service = svc
    return app


# Cross-origin isolation, which is what `SharedArrayBuffer` requires.
#
# The recorder's audio thread hands samples to the main thread through a
# lock-free ring buffer in shared memory. Without these two headers the
# constructor is simply absent and the capture engine falls back to copying
# every block through `postMessage` — which works, and allocates a buffer per
# render quantum on a real-time thread, which is exactly what the worklet was
# written to avoid.
#
# The cost is that cross-origin subresources must opt in via CORP. This
# interface loads nothing cross-origin: no CDN, no font service, no
# analytics. That was already true for other reasons and is what makes
# turning isolation on free here.
_ISOLATION_HEADERS = {
    "Cross-Origin-Opener-Policy": "same-origin",
    "Cross-Origin-Embedder-Policy": "require-corp",
}


class IsolationMiddleware(BaseHTTPMiddleware):
    """Apply the isolation headers to every response, not just the document.

    A worklet module fetched without `Cross-Origin-Resource-Policy` is
    blocked once the page is isolated, so the headers have to be consistent
    across the document, the modules and the audio.
    """

    async def dispatch(self, request, call_next):
        response = await call_next(request)
        for k, v in _ISOLATION_HEADERS.items():
            response.headers.setdefault(k, v)
        response.headers.setdefault("Cross-Origin-Resource-Policy", "same-origin")
        return response


app = create_app(os.environ.get("MIXENGINE_DATA", "./data"))
