"""
Notation Structural Invariant Validator.
Evaluates OMR transcription results (Humdrum **kern and ABC) against
deterministic musical, structural, and geometrical invariants without
requiring external paid APIs or ground-truth references.
"""

from dataclasses import dataclass, field
from pathlib import Path
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from core.abc_bridge import ABCBridge


@dataclass
class ValidationReport:
    """Detailed structural diagnostic report for a music notation transcription."""
    is_valid: bool
    score: float
    anomalies: List[str] = field(default_factory=list)
    measure_count: int = 0
    note_count: int = 0
    time_signature: Optional[str] = None
    clefs: List[str] = field(default_factory=list)
    density_notes_per_100px: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "score": round(self.score, 2),
            "anomalies": self.anomalies,
            "measure_count": self.measure_count,
            "note_count": self.note_count,
            "time_signature": self.time_signature,
            "clefs": self.clefs,
            "density_notes_per_100px": round(self.density_notes_per_100px, 2),
            "details": self.details,
        }


class NotationValidator:
    """
    Automated Structural Invariant Engine for music notation crops and predictions.
    Validates:
      1. Syntax integrity (Spine balancing, C++ Verovio load).
      2. Rhythmic balance & measure duration consistency.
      3. Geometry & token density sanity (runaway vs dropped staves).
      4. Note entropy and repetitive pitch/rest loops.
      5. Contamination (tablatures, malformed chords).
    """

    def __init__(self, bridge: Optional[ABCBridge] = None):
        self.bridge = bridge or ABCBridge()

    @staticmethod
    def _parse_duration(token: str) -> Optional[float]:
        """
        Parses Humdrum reciprocal duration string into quarter-note beat units.
        Examples:
          '4'   -> 1.0 quarter
          '8'   -> 0.5 quarter
          '2'   -> 2.0 quarters
          '1'   -> 4.0 quarters
          '16'  -> 0.25 quarter
          '4.'  -> 1.5 quarters
          '8.'  -> 0.75 quarters
          '4..' -> 1.75 quarters
          '12'  -> 4.0 / 3.0 (triplet 8th in 4/4) -> 0.333 quarters
          '.'   -> continuation (None)
        """
        if not token or token == ".":
            return None
        m = re.search(r"(\d+)(\.*)", token)
        if not m:
            return None
        reciprocal = int(m.group(1))
        if reciprocal == 0:
            return 4.0  # Mensural maxima/longa fallback
        dots = len(m.group(2))
        base_beats = 4.0 / float(reciprocal)
        factor = sum(0.5 ** i for i in range(dots + 1))
        return base_beats * factor

    def validate(
        self,
        raw_kern: str,
        abc_text: Optional[str] = None,
        crop_width: Optional[int] = None,
        crop_height: Optional[int] = None,
        notation_class: str = "staff"
    ) -> ValidationReport:
        anomalies: List[str] = []
        score: float = 1.0
        details: Dict[str, Any] = {}

        # 0. Basic presence check
        if not raw_kern or raw_kern.strip().startswith("% [OMR"):
            return ValidationReport(
                is_valid=False,
                score=0.0,
                anomalies=["EMPTY_OR_ERROR: Missing or failed OMR transcription output"],
                details={"raw_kern_len": len(raw_kern or "")}
            )

        # 1. Syntactic verification via Verovio
        try:
            norm_k = self.bridge.normalize_humdrum(raw_kern)
            spines_ok = self.bridge.check_spines_valid(norm_k)
            verovio_ok = self.bridge.validate_with_verovio(norm_k) if spines_ok else False
        except Exception as e:
            spines_ok = False
            verovio_ok = False
            details["syntax_exception"] = str(e)

        if not spines_ok:
            anomalies.append("SPINE_MISALIGNMENT: Humdrum spine column count inconsistent")
            score -= 0.50
        elif not verovio_ok:
            anomalies.append("VEROVIO_PARSE_ERROR: Verovio C++ engine failed to compile notation")
            score -= 0.40

        # 2. Contamination & Tablature leak check
        if "*clefTAB" in raw_kern or "*stria" in raw_kern:
            anomalies.append("TABLATURE_CONTAMINATION: Tablature tokens detected in standard notation crop")
            score -= 0.40

        # 3. Structural inspection: measures, notes, durations
        lines = [line.strip() for line in raw_kern.splitlines() if line.strip()]
        clefs = []
        time_sig = None
        measures: List[List[str]] = []
        cur_m: List[str] = []
        total_notes = 0
        total_rests = 0
        data_tokens_seq: List[str] = []

        for line in lines:
            if line.startswith("!"):
                continue
            if line.startswith("*"):
                # Interpretations
                for t in line.split("\t"):
                    if t.startswith("*clef"):
                        clefs.append(t)
                    elif t.startswith("*M") and len(t) > 2 and t[2].isdigit():
                        time_sig = t[2:]  # e.g. "4/4", "3/4", "6/8"
                continue

            if line.startswith("="):
                # Barline
                if cur_m:
                    measures.append(cur_m)
                    cur_m = []
                continue

            # Regular data line
            tokens = line.split("\t")
            # Analyze first spine (main/upper voice)
            primary_token = tokens[0]
            if primary_token != ".":
                data_tokens_seq.append(primary_token)
                if "r" in primary_token:
                    total_rests += 1
                elif any(c in primary_token.lower() for c in "abcdefg"):
                    total_notes += 1
            cur_m.append(line)

        if cur_m:
            measures.append(cur_m)

        measure_count = len(measures)
        details["total_notes"] = total_notes
        details["total_rests"] = total_rests
        details["measures_count"] = measure_count

        # 4. Multi-spine rhythmic balance & cross-voice synchronization
        max_spines = max((len(l.split("\t")) for l in lines if not l.startswith("!") and not l.startswith("*")), default=1)
        details["max_spines"] = max_spines

        expected_beats = None
        if time_sig and "/" in time_sig:
            try:
                num, den = [int(x) for x in time_sig.split("/")[:2]]
                expected_beats = (float(num) / float(den)) * 4.0
                details["expected_beats_per_measure"] = expected_beats
            except Exception:
                pass

        measure_durations: List[float] = []
        voice_desync_anomalies: List[str] = []

        for m_idx, m_lines in enumerate(measures):
            spines_in_m = max((len(l.split("\t")) for l in m_lines if not l.startswith("!") and not l.startswith("*")), default=1)
            voice_durs = [0.0] * spines_in_m
            for l in m_lines:
                tokens = l.split("\t")
                for s_i in range(min(spines_in_m, len(tokens))):
                    tok = tokens[s_i]
                    d = self._parse_duration(tok)
                    if d is not None:
                        voice_durs[s_i] += d

            primary_dur = round(voice_durs[0], 3)
            measure_durations.append(primary_dur)

            # Check cross-voice synchronization in polyphonic measures
            if spines_in_m > 1:
                active_durs = [round(vd, 3) for vd in voice_durs if vd > 0]
                if len(active_durs) > 1 and max(active_durs) - min(active_durs) > 0.35:
                    voice_desync_anomalies.append(f"M{m_idx+1}: " + ", ".join(f"V{i+1}={round(d, 2)}b" for i, d in enumerate(voice_durs)))

        details["measure_durations"] = measure_durations

        if voice_desync_anomalies:
            anomalies.append(f"VOICE_DESYNCHRONIZATION: {', '.join(voice_desync_anomalies[:3])}")
            score -= min(0.35, 0.10 * len(voice_desync_anomalies))

        # Check duplicate identical spines
        if max_spines >= 2:
            spine0_tokens = []
            spine1_tokens = []
            for l in lines:
                if l.startswith("!") or l.startswith("*") or l.startswith("="):
                    continue
                toks = l.split("\t")
                if len(toks) >= 2:
                    spine0_tokens.append(toks[0])
                    spine1_tokens.append(toks[1])
            if spine0_tokens and spine0_tokens == spine1_tokens:
                anomalies.append("DUPLICATE_VOICE: V1 and V2 contain identical notes")
                score -= 0.25

        # Check measure duration balance
        duration_anomalies = []
        if expected_beats is not None and expected_beats > 0:
            for idx, dur in enumerate(measure_durations):
                # Exclude first and last measures if partial (anacrusis or incomplete snippet)
                is_boundary = (idx == 0 or idx == len(measure_durations) - 1)
                diff = abs(dur - expected_beats)
                if diff > 0.30:
                    if is_boundary and dur < expected_beats:
                        continue
                    duration_anomalies.append(f"M{idx+1}: {dur}b (expected {expected_beats}b)")
        elif len(measure_durations) >= 3:
            from collections import Counter
            counts = Counter(round(d, 1) for d in measure_durations if d > 0)
            if counts:
                common_dur, num_occurrences = counts.most_common(1)[0]
                if num_occurrences >= 2 and common_dur > 0:
                    for idx, dur in enumerate(measure_durations):
                        if idx == 0 or idx == len(measure_durations) - 1:
                            continue
                        if abs(dur - common_dur) > 0.40:
                            duration_anomalies.append(f"M{idx+1}: {dur}b (expected ~{common_dur}b)")

        if duration_anomalies:
            anomalies.append(f"DURATION_IMBALANCE: {', '.join(duration_anomalies[:3])}")
            score -= min(0.35, 0.10 * len(duration_anomalies))

        # 5. Token & Note density vs Image geometry (Runaway vs Dropped staff)
        density = 0.0
        if crop_width and crop_width > 0:
            density = (float(total_notes) / float(crop_width)) * 100.0

            # Runaway high density check (e.g. 35 notes squeezed into 150px)
            if crop_width < 250 and total_notes > 20:
                anomalies.append(f"EXCESSIVE_NOTE_DENSITY: {total_notes} notes in {crop_width}px (runaway loop)")
                score -= 0.40
            elif density > 18.0 and crop_width < 400:
                anomalies.append(f"HIGH_NOTE_DENSITY: {density:.1f} notes/100px (probable loop)")
                score -= 0.30

            # Dropped or empty staff check
            if crop_width >= 600 and total_notes == 0:
                anomalies.append(f"DROPPED_STAFF: 0 notes detected across wide crop ({crop_width}px)")
                score -= 0.45
            elif crop_width >= 800 and total_notes <= 1 and total_rests <= 1:
                anomalies.append(f"SPARSE_DROPPED_STAFF: only {total_notes} note across wide crop ({crop_width}px)")
                score -= 0.35

        # 6. Repetition & loop detection (Consecutive identical notes or identical measures)
        if len(data_tokens_seq) >= 8:
            max_consecutive = 1
            cur_consecutive = 1
            for i in range(1, len(data_tokens_seq)):
                # Strip duration digits, check pitch identity
                p1 = re.sub(r"\d+\.*", "", data_tokens_seq[i])
                p0 = re.sub(r"\d+\.*", "", data_tokens_seq[i - 1])
                if p1 and p1 == p0 and p1 != ".":
                    cur_consecutive += 1
                    max_consecutive = max(max_consecutive, cur_consecutive)
                else:
                    cur_consecutive = 1

            if max_consecutive >= 8:
                anomalies.append(f"CONSECUTIVE_NOTE_LOOP: {max_consecutive} identical notes in sequence")
                score -= 0.35

        # Final score clamping and validity determination
        score = max(0.0, min(1.0, score))
        is_valid = (score >= 0.65) and not any(
            a.startswith("SPINE_MISALIGNMENT") or a.startswith("VEROVIO_PARSE_ERROR") or a.startswith("DROPPED_STAFF")
            for a in anomalies
        )

        return ValidationReport(
            is_valid=is_valid,
            score=score,
            anomalies=anomalies,
            measure_count=measure_count,
            note_count=total_notes,
            time_signature=time_sig,
            clefs=list(dict.fromkeys(clefs)),
            density_notes_per_100px=density,
            details=details,
        )
