"""
Command-line interface.

    python -m mixengine doctor
    python -m mixengine analyze-beats  --beats data/beats --out data/dna/beats
    python -m mixengine analyze-vocal  --vocal data/vocals/take.wav
    python -m mixengine match          --vocal data/vocals/take.wav
    python -m mixengine render         --vocal data/vocals/take.wav --beats 3
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Optional

from .config import CFG, Paths


def _setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-7s %(name)-22s %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )
    for noisy in ("numba", "matplotlib", "urllib3", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _load_metadata(path: Optional[str]) -> dict:
    """Load a metadata sidecar keyed by filename.

    Accepts either a dict keyed by filename, or a list of Mongo documents
    each carrying an `audio_file` field.
    """
    if not path or not os.path.exists(path):
        return {}
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, dict):
        return data
    if isinstance(data, list):
        out = {}
        for doc in data:
            key = doc.get("audio_file") or doc.get("filename")
            if key:
                out[key] = doc
        return out
    return {}


# ─────────────────────────────────────────────────────────────────────────────
# Commands
# ─────────────────────────────────────────────────────────────────────────────

def cmd_doctor(args) -> int:
    from .core.capabilities import CAPS
    print(CAPS.summary())
    print(f"  quality tier: {CAPS.tier}")
    print("  analysis backends: " + ", ".join(
        f"{k}={v}" for k, v in CAPS.analysis_backends().items()))
    print()
    if not CAPS.can_render:
        print("  BLOCKED: install librosa and soundfile to proceed.")
        return 1
    missing = []
    if not CAPS.can_stretch_well:
        missing.append("pyrubberband + the rubberband binary (formant-preserving "
                       "stretch/shift -- biggest single quality win)")
    if not CAPS.madmom:
        missing.append("madmom (downbeat tracking -- needed for bar-accurate alignment)")
    if not CAPS.can_separate:
        missing.append("demucs (stems -- enables drum-safe pitch shift and ducking)")
    if not CAPS.torchcrepe:
        missing.append("torchcrepe (accurate f0 -- improves key detection and tuning)")
    if not CAPS.pedalboard:
        missing.append("pedalboard (reverb and limiter)")
    if not CAPS.pyloudnorm:
        missing.append("pyloudnorm (LUFS metering)")

    if missing:
        print("  Recommended additions, in order of impact:")
        for m in missing:
            print(f"    - {m}")
    else:
        print("  All recommended components present.")
    return 0


def cmd_analyze_beats(args) -> int:
    from .analysis import beat_dna
    meta = _load_metadata(args.metadata)
    if meta:
        print(f"Loaded metadata for {len(meta)} beats")
    results = beat_dna.extract_catalog(
        beats_dir=args.beats, out_dir=args.out,
        metadata_by_file=meta, stems_dir=args.stems,
        do_separation=not args.no_separation,
        skip_existing=not args.force)

    ok = [r for r in results if r.get("status") == "ok"]
    print(f"\nAnalysed {len(ok)}/{len(results)} beats")
    if ok:
        print(f"{'ID':<22}{'BPM':>7}  {'KEY':<12}{'CAM':>4}  {'POCKET':>7}  GENRE")
        print("-" * 74)
        for r in ok:
            key = (r.get("key") or {}).get("name", "atonal")
            print(f"{r['beat_id']:<22}{r.get('bpm', 0):>7.1f}  {key:<12}"
                  f"{r.get('camelot') or '-':>4}  {r.get('pocket_score', 0):>7.2f}  "
                  f"{r.get('genre') or '-'}")

        n_bpm_mismatch = sum(1 for r in ok
                             if r.get("bpm_tagged") and not r.get("bpm_verified"))
        n_key_mismatch = sum(1 for r in ok
                             if r.get("key_tagged") and not r.get("key_verified"))
        tagged = sum(1 for r in ok if r.get("bpm_tagged"))
        if tagged:
            print(f"\nProducer tag accuracy: "
                  f"BPM {tagged - n_bpm_mismatch}/{tagged} verified, "
                  f"key {tagged - n_key_mismatch}/{tagged} verified")
    return 0


def cmd_analyze_vocal(args) -> int:
    from .analysis import vocal_dna
    out_dir = args.out or CFG.paths.vocal_dna
    os.makedirs(out_dir, exist_ok=True)
    ref = None
    if args.recorded_over:
        from .analysis import beat_dna
        catalog = beat_dna.load_catalog(args.dna or CFG.paths.beat_dna)
        ref = next((b for b in catalog if b.get("beat_id") == args.recorded_over), None)
        if ref is None:
            print(f"warning: beat {args.recorded_over!r} not found in the catalog; "
                  f"analysing without a reference beat")
    dna = vocal_dna.extract(args.vocal, user_bpm=args.bpm, user_key=args.key,
                            reference_beat_path=(ref or {}).get("source_path"),
                            reference_beat_dna=ref)
    path = vocal_dna.save(dna, out_dir)

    print(f"\n{dna.get('summary', '')}\n")
    req = dna.get("beat_requirements", {})
    print("Beat requirements:")
    print(f"  tempo:  {req.get('tempo_range')}"
          f"{' (half-time: ' + str(req.get('tempo_range_halftime')) + ')' if req.get('tempo_range_halftime') else ''}")
    print(f"  keys:   {', '.join(req.get('compatible_key_names', [])[:5])}")
    print(f"  genres: {', '.join(req.get('genres', [])[:5])}")
    print(f"  pocket: >= {req.get('min_pocket_score')}")
    for w in dna.get("warnings", []):
        print(f"  ! {w}")
    print(f"\nSaved: {path}")
    return 0


def cmd_match(args) -> int:
    from .analysis import beat_dna
    from .audio import pipeline
    catalog = beat_dna.load_catalog(args.dna or CFG.paths.beat_dna)
    if not catalog:
        print(f"No beat DNA found in {args.dna or CFG.paths.beat_dna}. "
              f"Run 'analyze-beats' first.")
        return 1

    res = pipeline.analyze_only(args.vocal, catalog, n=args.n,
                                user_bpm=args.bpm, user_key=args.key)
    if res.get("status") != "ok":
        print(f"Failed: {res.get('error')}")
        return 1

    print(f"\n{res['vocal_summary']}\n")
    report = res["match_report"]
    print(report["message"])
    print(f"(catalog {report['catalog_size']}, "
          f"considered {report['candidates_considered']}, "
          f"relaxation level {report['relaxation_level']})\n")

    for i, m in enumerate(report["matches"], 1):
        print(f"{i}. {m['title'] or m['beat_id']}  --  {m['score_pct']}%")
        print(f"   {m['transform_summary']}")
        for r in m["reasons"][:3]:
            print(f"   + {r}")
        for w in m["warnings"][:2]:
            print(f"   ! {w}")
        print()
    return 0


def cmd_render(args) -> int:
    from .analysis import beat_dna
    from .audio import pipeline
    from .core.intents import Intents
    catalog = beat_dna.load_catalog(args.dna or CFG.paths.beat_dna)
    if not catalog:
        print("No beat DNA found. Run 'analyze-beats' first.")
        return 1

    out = pipeline.run(
        vocal_path=args.vocal, catalog=catalog,
        out_dir=args.out or CFG.paths.outputs,
        n_beats=args.beats, variants_per_beat=args.variants,
        user_bpm=args.bpm, user_key=args.key,
        beat_ids=args.beat_id,
        intents=Intents.from_dict({
            "bpm": args.bpm, "key": args.key,
            "nudge": getattr(args, "nudge", None),
            "performance": getattr(args, "performance", None),
            "lead_in": getattr(args, "lead_in", None),
            "entry": getattr(args, "entry", None),
            "noise": getattr(args, "noise", None),
            "length": getattr(args, "length", None),
            "voice": getattr(args, "voice", None)}))

    if out["status"] == "needs_answers":
        # The dashboard puts these as questions with the engine's own
        # default preselected. Here they are flags, so say which one.
        print("\nThis take is not rendered until you answer:")
        for q in out.get("questions") or []:
            print("\n  %s" % q["text"])
            if q.get("reason"):
                print("  (%s)" % q["reason"])
            for opt in q["options"]:
                print("    --%-12s %-10s %s"
                      % (q["intent"].replace("_", "-"), opt["value"],
                         opt["label"]))
        return 1

    if out["status"] != "ok":
        print(f"\n{out.get('message') or out.get('error')}")
        return 1

    print(f"\n{out['vocal_summary']}")
    print(f"{out['match_report']['message']}\n")
    print(f"{'#':<3}{'VARIANT':<12}{'BEAT':<26}{'SCORE':>7}{'PASS':>6}  FILE")
    print("-" * 92)
    for i, r in enumerate(out["renders"], 1):
        passed = "yes" if r["critic"].get("passed") else "no"
        print(f"{i:<3}{r['variant']:<12}{r['beat_title'][:25]:<26}"
              f"{r['score_pct']:>6}%{passed:>6}  {os.path.basename(r['path'])}")
    print(f"\nTotal: {out['total_seconds']:.0f}s   Output: {args.out or CFG.paths.outputs}")
    return 0


def cmd_serve(args) -> int:
    """Start the local interface.

    Binds to loopback by default. The server exposes the filesystem paths it
    was given and runs analysis on whatever is uploaded, so it is a local
    tool rather than something to put on a public interface without putting
    authentication in front of it first.
    """
    try:
        import uvicorn
    except ImportError:
        print("The web interface needs extra packages:\n"
              "    pip install 'mixengine[web]'")
        return 1
    from .api.app import create_app

    root = args.data or CFG.paths.root
    os.environ["MIXENGINE_DATA"] = root
    print(f"mixengine  ->  http://{args.host}:{args.port}   (data: {os.path.abspath(root)})")
    uvicorn.run(create_app(root), host=args.host, port=args.port,
                log_level="warning")
    return 0


# ─────────────────────────────────────────────────────────────────────────────

def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="mixengine",
        description="Vocal + beat -> studio-ready song. "
                    "Analyse a catalog, match a vocal to it, render masters.")
    p.add_argument("-v", "--verbose", action="store_true")
    p.add_argument("--root", default=None, help="data root (default ./data)")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("doctor", help="report installed capabilities").set_defaults(
        func=cmd_doctor)

    ab = sub.add_parser("analyze-beats", help="build beat DNA for a catalog")
    ab.add_argument("--beats", default=None, help="directory of beat audio files")
    ab.add_argument("--out", default=None, help="where to write DNA json")
    ab.add_argument("--stems", default=None, help="where to write separated stems")
    ab.add_argument("--metadata", default=None,
                    help="json sidecar: {filename: mongo_doc} or a list of docs")
    ab.add_argument("--no-separation", action="store_true",
                    help="skip stem separation (much faster, lower quality renders)")
    ab.add_argument("--force", action="store_true", help="re-analyse cached beats")
    ab.set_defaults(func=cmd_analyze_beats)

    av = sub.add_parser("analyze-vocal", help="build vocal DNA")
    av.add_argument("--vocal", required=True)
    av.add_argument("--out", default=None)
    av.add_argument("--bpm", type=float, default=None, help="known vocal BPM")
    av.add_argument("--key", default=None, help="known vocal key, e.g. f#_minor")
    av.add_argument("--recorded-over", default=None, metavar="BEAT_ID",
                    help="catalog beat that was playing during the take: "
                         "enables bleed cancellation and fixes the tempo")
    av.add_argument("--dna", default=None, help="beat DNA directory")
    av.set_defaults(func=cmd_analyze_vocal)

    mt = sub.add_parser("match", help="find the best beats for a vocal")
    mt.add_argument("--vocal", required=True)
    mt.add_argument("--dna", default=None)
    mt.add_argument("-n", type=int, default=5)
    mt.add_argument("--bpm", type=float, default=None)
    mt.add_argument("--key", default=None)
    mt.set_defaults(func=cmd_match)

    rd = sub.add_parser("render", help="match and render full songs")
    rd.add_argument("--vocal", required=True)
    rd.add_argument("--dna", default=None)
    rd.add_argument("--out", default=None)
    rd.add_argument("--beats", type=int, default=3, help="how many beats to use")
    rd.add_argument("--variants", type=int, default=1,
                    help="mix variants per beat (1-5)")
    rd.add_argument("--beat-id", nargs="*", default=None,
                    help="force specific beat ids instead of matching")
    rd.add_argument("--bpm", type=float, default=None)
    rd.add_argument("--key", default=None)
    rd.add_argument("--nudge", type=float, default=None,
                    help="shift the vocal by N beats (+ later, - earlier). "
                         "Where a vocal's bars sit against a beat it was "
                         "not recorded to is genuinely ambiguous; this is "
                         "the last word.")
    # The answers the dashboard collects as questions. Without these a
    # take the analysis blocked could not be rendered from here at all.
    rd.add_argument("--performance", default=None,
                    choices=("rap", "melodic_rap", "sung", "spoken"),
                    help="what the take is, when the classifier was unsure")
    rd.add_argument("--lead-in", default=None, choices=("trim", "keep"),
                    dest="lead_in",
                    help="cut or keep the sound before the first line")
    rd.add_argument("--entry", default=None, choices=("section", "top"),
                    help="bring the vocal in at the beat's first section or "
                         "at the top of the file")
    rd.add_argument("--noise", default=None, choices=("accept", "rerecord"),
                    help="accept: render a take whose noise could not be "
                         "removed")
    rd.add_argument("--length", default=None, choices=("accept", "rerecord"),
                    help="accept: render a take too short to build a song "
                         "from")
    rd.add_argument("--voice", default=None, choices=("accept", "rerecord"),
                    help="accept: use a file with no voice in it as the vocal")
    rd.set_defaults(func=cmd_render)

    sv = sub.add_parser("serve", help="run the local web interface")
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--port", type=int, default=8000)
    sv.add_argument("--data", default=None, help="data root (default ./data)")
    sv.set_defaults(func=cmd_serve)

    args = p.parse_args(argv)
    _setup_logging(args.verbose)

    if args.root:
        CFG.paths = Paths(root=args.root)
    for attr in ("beats", "out", "stems", "dna"):
        if getattr(args, attr, None) is None and hasattr(args, attr):
            defaults = {"beats": CFG.paths.beats, "dna": CFG.paths.beat_dna,
                        "stems": CFG.paths.stems}
            if attr in defaults:
                setattr(args, attr, defaults[attr])
    if getattr(args, "out", None) is None and args.command == "analyze-beats":
        args.out = CFG.paths.beat_dna

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
