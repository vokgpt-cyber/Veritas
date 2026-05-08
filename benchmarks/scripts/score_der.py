"""Compute DER + JER + speaker-counting error between a hypothesis RTTM and a
reference RTTM using pyannote.metrics.

Usage:
    python benchmarks/scripts/score_der.py \\
        --ref benchmarks/gold/court_hearing_129.rttm \\
        --hyp benchmarks/diar/pyannote_community1/court_hearing_129.rttm \\
        --out benchmarks/metrics/diar_pyannote_community1_court_hearing_129.json \\
        --tag pyannote_community1 \\
        --collar 0.25

Metrics reported:
    der, der_skip_overlap, jer, miss, false_alarm, confusion,
    ref_total_speech_s, hyp_total_speech_s,
    n_ref_speakers, n_hyp_speakers, speaker_count_error

Dependencies:
    pyannote.metrics >= 4.0
    pyannote.core    >= 5.0
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pyannote.core import Annotation, Segment  # type: ignore
from pyannote.database.util import load_rttm  # type: ignore
from pyannote.metrics.diarization import DiarizationErrorRate, JaccardErrorRate  # type: ignore


def load_rttm_annotation(path: Path) -> Annotation:
    data = load_rttm(str(path))
    if not data:
        raise ValueError(f"no RTTM entries in {path}")
    # load_rttm returns dict[recording_id -> Annotation]
    key = next(iter(data))
    return data[key]


def score(ref: Annotation, hyp: Annotation, collar: float) -> dict:
    der_metric = DiarizationErrorRate(collar=collar, skip_overlap=False)
    der_value = der_metric(ref, hyp, detailed=True)
    der_skip = DiarizationErrorRate(collar=collar, skip_overlap=True)(ref, hyp)
    jer = JaccardErrorRate(collar=collar)(ref, hyp)

    # Total speech time
    ref_speech = sum(seg.duration for seg in ref.get_timeline())
    hyp_speech = sum(seg.duration for seg in hyp.get_timeline())
    n_ref_speakers = len(ref.labels())
    n_hyp_speakers = len(hyp.labels())

    return {
        "der": round(float(der_value["diarization error rate"]), 4),
        "der_skip_overlap": round(float(der_skip), 4),
        "jer": round(float(jer), 4),
        "miss": round(float(der_value["missed detection"]), 4),
        "false_alarm": round(float(der_value["false alarm"]), 4),
        "confusion": round(float(der_value["confusion"]), 4),
        "total_s": round(float(der_value["total"]), 2),
        "correct_s": round(float(der_value["correct"]), 2),
        "ref_total_speech_s": round(ref_speech, 2),
        "hyp_total_speech_s": round(hyp_speech, 2),
        "n_ref_speakers": n_ref_speakers,
        "n_hyp_speakers": n_hyp_speakers,
        "speaker_count_error": n_hyp_speakers - n_ref_speakers,
        "collar_s": collar,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, type=Path)
    ap.add_argument("--hyp", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--tag", default="")
    ap.add_argument("--collar", type=float, default=0.25,
                    help="DER forgiveness collar around reference turn boundaries, seconds")
    args = ap.parse_args()

    ref = load_rttm_annotation(args.ref)
    hyp = load_rttm_annotation(args.hyp)
    metrics = score(ref, hyp, args.collar)
    metrics["ref_file"] = str(args.ref)
    metrics["hyp_file"] = str(args.hyp)
    if args.tag:
        metrics["tag"] = args.tag
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nWritten to {args.out}")


if __name__ == "__main__":
    main()
