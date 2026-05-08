"""Transcript aligner for combining ASR transcription with speaker diarization.

Alignment strategy (2026-04-21):
  1. Pre-split ASR segments at sentence boundaries (. ! ?) so long GigaAM
     chunks (15-22s) cannot swallow multiple speaker turns. Uses word-level
     timestamps when available, falls back to proportional-by-char otherwise.
  2. For each post-split ASR segment, sum overlap time per speaker across
     ALL diarization segments and assign the speaker with maximum aggregate
     time. Previous implementation picked the single diarization segment with
     largest overlap, which is systematically wrong when one speaker's time
     is fragmented across multiple short turns inside a long ASR chunk.
  3. For court hearings, optionally split ASR chunks at diarization speaker
     boundaries before final assignment. This preserves short courtroom Q/A
     turns that punctuation-only splitting misses.

Sets AlignedSegment.attribution_confidence = winner_time / total_time to
flag segments where multiple speakers had substantial presence.
"""
from __future__ import annotations

import logging
import re
from collections import defaultdict
from typing import Optional

from backend.app.models import (
    AlignedSegment,
    DiarizationSegment,
    Participant,
    TranscriptionSegment,
    WordInfo,
)

logger = logging.getLogger(__name__)

_SENTENCE_SPLIT_RE = re.compile(r"([\.!\?]+[\"'\)\]\u00bb]?)\s+")
SENTENCE_SPLIT_MIN_DURATION_S = 6.0
MIN_SPLIT_PIECE_DURATION_S = 0.4
DIARIZATION_SPLIT_MIN_DURATION_S = 1.2
DIARIZATION_SPLIT_MIN_RUN_DURATION_S = 0.25
DIARIZATION_SPLIT_MERGE_GAP_S = 0.12


class TranscriptAligner:
    """Aligns transcription segments with diarization to produce 'who said what'."""

    @staticmethod
    def _overlap_duration(s1a: float, s1b: float, s2a: float, s2b: float) -> float:
        return max(0.0, min(s1b, s2b) - max(s1a, s2a))

    @staticmethod
    def _find_sentence_cut_points(text: str) -> list[int]:
        return [m.end() for m in _SENTENCE_SPLIT_RE.finditer(text)]

    @staticmethod
    def _time_slice_from_words(full_text, start_char, end_char, words, fallback_start, fallback_end):
        try:
            cursor = 0
            sub = []
            for w in words:
                tok = w.word.strip()
                if not tok:
                    continue
                idx = full_text.find(tok, cursor)
                if idx < 0:
                    continue
                tok_end = idx + len(tok)
                cursor = tok_end
                if idx >= end_char:
                    break
                if tok_end <= start_char:
                    continue
                sub.append(w)
            if not sub:
                return fallback_start, fallback_end, []
            sub_start = min(float(w.start) for w in sub)
            sub_end = max(float(w.end) for w in sub)
            if sub_end <= sub_start:
                return fallback_start, fallback_end, []
            return sub_start, sub_end, sub
        except Exception as exc:
            logger.debug(f"word-timing slice fell back: {exc}")
            return fallback_start, fallback_end, []

    @classmethod
    def _split_one_segment(cls, trans_seg):
        duration = trans_seg.end - trans_seg.start
        if duration < SENTENCE_SPLIT_MIN_DURATION_S:
            return [trans_seg]
        text = trans_seg.text
        cuts = cls._find_sentence_cut_points(text)
        if not cuts:
            return [trans_seg]
        slice_bounds = []
        prev = 0
        for c in cuts:
            slice_bounds.append((prev, c))
            prev = c
        slice_bounds.append((prev, len(text)))
        slice_bounds = [(a, b) for (a, b) in slice_bounds if text[a:b].strip()]
        if len(slice_bounds) <= 1:
            return [trans_seg]

        words = trans_seg.words or []
        have_wt = len(words) > 0
        pieces = []
        for (a, b) in slice_bounds:
            sub_text = text[a:b].strip()
            if not sub_text:
                continue
            if have_wt:
                t0, t1, sw = cls._time_slice_from_words(text, a, b, words, trans_seg.start, trans_seg.end)
                if t1 <= t0:
                    t0 = trans_seg.start + duration * (a / max(1, len(text)))
                    t1 = trans_seg.start + duration * (b / max(1, len(text)))
                    sw = []
            else:
                t0 = trans_seg.start + duration * (a / max(1, len(text)))
                t1 = trans_seg.start + duration * (b / max(1, len(text)))
                sw = []
            if (t1 - t0) < MIN_SPLIT_PIECE_DURATION_S and pieces:
                last = pieces[-1]
                last.end = max(last.end, t1, last.start)
                last.text = (last.text + " " + sub_text).strip()
                if sw:
                    last.words = list(last.words) + sw
                continue
            pieces.append(TranscriptionSegment(
                start=t0, end=t1, text=sub_text,
                confidence=trans_seg.confidence, words=sw,
            ))
        if not pieces:
            return [trans_seg]
        for p in pieces:
            p.start = max(trans_seg.start, p.start)
            p.end = min(trans_seg.end, p.end)
            if p.end < p.start:
                p.end = p.start
        pieces[0].start = trans_seg.start
        pieces[-1].end = trans_seg.end
        for p in pieces:
            if p.end < p.start:
                p.end = p.start
        return pieces

    @classmethod
    def split_transcription_by_sentences(cls, transcription):
        if not transcription:
            return transcription
        out = []
        split_count = 0
        for seg in transcription:
            pieces = cls._split_one_segment(seg)
            if len(pieces) > 1:
                split_count += 1
            out.extend(pieces)
        if split_count:
            logger.info(f"Sentence-split: {split_count}/{len(transcription)} ASR segments split into {len(out)} sub-segments")
        return out

    @classmethod
    def _speaker_runs_for_window(cls, start: float, end: float, diarization):
        """Return dominant diarization speaker runs within an ASR window.

        The pyannote output can contain overlapping tracks. We build small
        non-overlapping intervals from all diarization boundaries, choose the
        speaker with the largest overlap inside each interval, then merge
        adjacent intervals with the same speaker. This gives the court splitter
        stable boundaries without requiring pyannote's exclusive output.
        """
        boundaries = {start, end}
        candidates = []
        for ds in diarization:
            ov_start = max(start, float(ds.start))
            ov_end = min(end, float(ds.end))
            if ov_end <= ov_start:
                continue
            candidates.append(ds)
            boundaries.add(ov_start)
            boundaries.add(ov_end)
        if not candidates:
            return []

        points = sorted(boundaries)
        raw_runs: list[tuple[float, float, str]] = []
        for a, b in zip(points, points[1:]):
            if b - a < 0.01:
                continue
            per = defaultdict(float)
            for ds in candidates:
                ov = cls._overlap_duration(a, b, ds.start, ds.end)
                if ov > 0:
                    per[ds.speaker_id] += ov
            if not per:
                continue
            sid, _ = max(per.items(), key=lambda kv: kv[1])
            raw_runs.append((a, b, sid))

        if not raw_runs:
            return []

        merged: list[tuple[float, float, str]] = []
        for a, b, sid in raw_runs:
            if (
                merged
                and merged[-1][2] == sid
                and a - merged[-1][1] <= DIARIZATION_SPLIT_MERGE_GAP_S
            ):
                prev_a, _, _ = merged[-1]
                merged[-1] = (prev_a, b, sid)
            else:
                merged.append((a, b, sid))

        return [
            (a, b, sid)
            for a, b, sid in merged
            if b - a >= DIARIZATION_SPLIT_MIN_RUN_DURATION_S
        ]

    @staticmethod
    def _cut_text_by_time_proportion(text: str, start: float, end: float, cut: float) -> int:
        if end <= start or not text:
            return len(text)
        ratio = min(1.0, max(0.0, (cut - start) / (end - start)))
        approx = int(round(len(text) * ratio))
        if approx <= 0 or approx >= len(text):
            return approx
        # Prefer whitespace within a small window around the proportional cut.
        window = max(8, min(40, len(text) // 8))
        lo = max(1, approx - window)
        hi = min(len(text) - 1, approx + window)
        best = None
        best_dist = None
        for idx in range(lo, hi + 1):
            if text[idx].isspace():
                dist = abs(idx - approx)
                if best is None or dist < best_dist:
                    best = idx
                    best_dist = dist
        return best if best is not None else approx

    @classmethod
    def _split_one_segment_by_diarization(cls, trans_seg, diarization):
        duration = trans_seg.end - trans_seg.start
        if duration < DIARIZATION_SPLIT_MIN_DURATION_S:
            return [trans_seg]

        runs = cls._speaker_runs_for_window(
            float(trans_seg.start), float(trans_seg.end), diarization
        )
        speaker_changes = sum(
            1 for i in range(1, len(runs)) if runs[i][2] != runs[i - 1][2]
        )
        if speaker_changes == 0:
            return [trans_seg]

        words = list(trans_seg.words or [])
        pieces = []
        if words:
            for run_start, run_end, _sid in runs:
                sub_words = [
                    w for w in words
                    if run_start <= ((float(w.start) + float(w.end)) / 2.0) < run_end
                ]
                if not sub_words:
                    continue
                sub_text = " ".join(w.word.strip() for w in sub_words if w.word.strip())
                if not sub_text:
                    continue
                pieces.append(TranscriptionSegment(
                    start=max(trans_seg.start, run_start),
                    end=min(trans_seg.end, run_end),
                    text=sub_text.strip(),
                    confidence=trans_seg.confidence,
                    words=sub_words,
                ))
        else:
            text = trans_seg.text or ""
            text_start = 0
            for idx, (run_start, run_end, _sid) in enumerate(runs):
                if idx == len(runs) - 1:
                    text_end = len(text)
                else:
                    text_end = cls._cut_text_by_time_proportion(
                        text, trans_seg.start, trans_seg.end, run_end
                    )
                sub_text = text[text_start:text_end].strip()
                text_start = text_end
                if not sub_text:
                    continue
                pieces.append(TranscriptionSegment(
                    start=max(trans_seg.start, run_start),
                    end=min(trans_seg.end, run_end),
                    text=sub_text,
                    confidence=trans_seg.confidence,
                    words=[],
                ))

        if len(pieces) <= 1:
            return [trans_seg]
        pieces[0].start = trans_seg.start
        pieces[-1].end = trans_seg.end
        return pieces

    @classmethod
    def split_transcription_by_diarization(cls, transcription, diarization):
        if not transcription or not diarization:
            return transcription
        out = []
        split_count = 0
        for seg in transcription:
            pieces = cls._split_one_segment_by_diarization(seg, diarization)
            if len(pieces) > 1:
                split_count += 1
            out.extend(pieces)
        if split_count:
            logger.info(
                "Diarization-split: %d/%d ASR segments split into %d sub-segments",
                split_count,
                len(transcription),
                len(out),
            )
        return out

    @classmethod
    def align(
        cls,
        transcription,
        diarization,
        sentence_split: bool = True,
        diarization_split: bool = False,
    ):
        if not transcription:
            logger.warning("Empty transcription provided")
            return []
        if sentence_split:
            transcription = cls.split_transcription_by_sentences(transcription)
        if diarization_split and diarization:
            transcription = cls.split_transcription_by_diarization(
                transcription, diarization
            )
        logger.info(
            "Aligning %d trans segs with %d dia segs "
            "(sentence_split=%s, diarization_split=%s)",
            len(transcription),
            len(diarization),
            sentence_split,
            diarization_split,
        )

        if not diarization:
            logger.warning("Empty diarization; assigning to speaker_0")
            return [AlignedSegment(
                start=min(float(s.start), float(s.end)),
                end=max(float(s.start), float(s.end)),
                text=s.text,
                speaker_id="speaker_0", speaker_name=None,
                confidence=s.confidence, attribution_confidence=None,
            ) for s in transcription]

        aligned = []
        for ts in transcription:
            ts_start = float(ts.start)
            ts_end = float(ts.end)
            if ts_end < ts_start:
                logger.debug(
                    "Normalizing inverted ASR segment bounds: %.3f-%.3f",
                    ts_start,
                    ts_end,
                )
                ts_start, ts_end = ts_end, ts_start

            per = defaultdict(float)
            names = {}
            overlap_time = 0.0  # T2.2: total seconds where multiple
                                # speakers were active inside this ASR
                                # window (interruption / talking-over).
            for ds in diarization:
                ov = cls._overlap_duration(ts_start, ts_end, ds.start, ds.end)
                if ov > 0:
                    per[ds.speaker_id] += ov
                    if ds.speaker_name and ds.speaker_id not in names:
                        names[ds.speaker_id] = ds.speaker_name
                    if getattr(ds, "is_overlap", False):
                        overlap_time += ov

            best = None
            best_name = None
            conf = None
            if per:
                best, win = max(per.items(), key=lambda kv: kv[1])
                best_name = names.get(best)
                tot = sum(per.values())
                conf = round(win / tot, 3) if tot > 0 else None

                # T2.2 (2026-04-23): if a sizeable fraction of this ASR
                # window was a multi-speaker overlap region, downgrade
                # the attribution confidence so the editor UI can flag
                # it amber/red. Stakeholder explicitly called out
                # interruption mis-attribution as a problem.
                if conf is not None and tot > 0 and overlap_time > 0:
                    overlap_share = overlap_time / tot
                    if overlap_share >= 0.10:  # at least 10% overlap
                        # Cap confidence at 0.55 so it lands in the
                        # "contested" bucket per UI thresholds.
                        conf = round(min(conf, 0.55), 3)
            else:
                mid = (ts_start + ts_end) / 2.0
                min_d = float("inf")
                for ds in diarization:
                    dmid = (ds.start + ds.end) / 2.0
                    d = abs(mid - dmid)
                    if d < min_d:
                        min_d = d
                        best = ds.speaker_id
                        best_name = ds.speaker_name
                logger.debug(f"No overlap [{ts_start:.2f}-{ts_end:.2f}], nearest={best}")

            aligned.append(AlignedSegment(
                start=ts_start, end=ts_end, text=ts.text,
                speaker_id=best, speaker_name=best_name,
                confidence=ts.confidence, attribution_confidence=conf,
            ))

        if aligned:
            confs = [a.attribution_confidence for a in aligned if a.attribution_confidence is not None]
            if confs:
                low = sum(1 for c in confs if c < 0.6)
                avg = sum(confs) / len(confs)
                logger.info(f"Alignment confidence: avg={avg:.2f}, {low}/{len(confs)} below 0.6 (contested)")

        logger.info(f"Alignment complete: {len(aligned)} aligned segments")
        return aligned

    @staticmethod
    def merge_consecutive(segments):
        if not segments:
            return []
        logger.info(f"Merging from {len(segments)} segments")
        merged = []
        cur = AlignedSegment(**segments[0].model_dump())
        for seg in segments[1:]:
            if seg.speaker_id == cur.speaker_id and seg.speaker_name == cur.speaker_name:
                cur.end = seg.end
                cur.text += " " + seg.text
                cur.confidence = (cur.confidence + seg.confidence) / 2
                if cur.attribution_confidence is not None and seg.attribution_confidence is not None:
                    cur.attribution_confidence = min(cur.attribution_confidence, seg.attribution_confidence)
                elif seg.attribution_confidence is not None:
                    cur.attribution_confidence = seg.attribution_confidence
            else:
                merged.append(cur)
                cur = AlignedSegment(**seg.model_dump())
        merged.append(cur)
        logger.info(f"Merged to {len(merged)} segments")
        return merged

    @staticmethod
    def get_participants(segments):
        logger.debug(f"Extracting participants from {len(segments)} segments")
        sp = {}
        total = 0.0
        for s in segments:
            d = s.end - s.start
            total += d
            if s.speaker_id not in sp:
                sp[s.speaker_id] = {"duration": 0.0, "name": s.speaker_name}
            sp[s.speaker_id]["duration"] += d
        out = []
        for sid, data in sp.items():
            d = data["duration"]
            share = (d / total * 100) if total > 0 else 0.0
            out.append(Participant(speaker_id=sid, speaker_name=data["name"], speaking_time=d, speaking_share=round(share, 1)))
        out.sort(key=lambda p: p.speaking_time, reverse=True)
        logger.info(f"Extracted {len(out)} unique participants")
        return out

    @classmethod
    def validate_alignment(cls, segments):
        issues = []
        if not segments:
            return False, ["Empty segment list"]
        prev_end = segments[0].start
        for i, seg in enumerate(segments):
            if seg.start < prev_end:
                issues.append(f"Segment {i}: start ({seg.start:.2f}) < previous end ({prev_end:.2f})")
            if seg.end < seg.start:
                issues.append(f"Segment {i}: end ({seg.end:.2f}) < start ({seg.start:.2f})")
            if not seg.speaker_id:
                issues.append(f"Segment {i}: missing speaker_id")
            prev_end = seg.end
        return len(issues) == 0, issues
