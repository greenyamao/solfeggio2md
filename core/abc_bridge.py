"""
ABC Bridge: Robust Music Notation Translation and Validation
Utilizes Verovio C++ engine for structural validation and rendering,
coupled with Willem Vree's xml2abc for pristine multi-voice ABC generation.
"""

from pathlib import Path
import contextlib
import io
import logging
import re
import sys
from typing import Optional, Tuple, Any
import music21
import verovio
from abc_xml_converter import convert_xml2abc

# Silence verbose music21 terminal warnings (e.g. unknown clef types, unterminated spines, spineParser)
try:
    from music21.environment import Environment
    Environment.warn = lambda self, *args, **kwargs: None
    Environment.printDebug = lambda self, *args, **kwargs: None
except Exception:
    pass

for logger_name in ("music21", "music21.humdrum", "music21.musicxml", "humdrum", "humdrum.spineParser"):
    logging.getLogger(logger_name).setLevel(logging.ERROR)


class ABCBridge:
    def __init__(self):
        try:
            verovio.enableLog(verovio.LOG_OFF)
        except Exception:
            pass
        self._tk = verovio.toolkit()

        # Suppress third-party MIDI channel and syntax warning spam from music21
        for logger_name in ("music21", "music21.humdrum", "music21.musicxml", "humdrum", "humdrum.spineParser"):
            logging.getLogger(logger_name).setLevel(logging.ERROR)

    @staticmethod
    def check_spines_valid(norm_text: str) -> bool:
        """
        Pure Python mathematical verification of Humdrum spine consistency.
        Guarantees that every line matches the active column count before passing to C++ Verovio.
        """
        lines = [l.strip() for l in norm_text.splitlines() if l.strip() and not l.startswith("!")]
        if not lines or not any(l.startswith("**") for l in lines):
            return False
        s = len(lines[0].split("\t"))
        for line in lines[1:]:
            tokens = line.split("\t")
            if len(tokens) != s:
                return False
            if any(t in ("*^", "*v") for t in tokens):
                if not all(t.startswith("*") for t in tokens):
                    return False
                next_count = 0
                i = 0
                while i < len(tokens):
                    if tokens[i] == "*^":
                        next_count += 2
                        i += 1
                    elif tokens[i] == "*v":
                        next_count += 1
                        while i < len(tokens) and tokens[i] == "*v":
                            i += 1
                    else:
                        next_count += 1
                        i += 1
                s = max(1, next_count)
        return True

    @staticmethod
    def sanitize_runaway_kern(raw_kern: str, max_repeats: int = 2) -> str:
        """
        Detects and truncates degenerative runaway loops and attention wrap-arounds in Humdrum **kern:
        1. Truncates immediately at any terminal barline (==, =||, =:|, =:|!, *-).
        2. Detects header re-occurrence: if *clef, *k[, *M, or **kern occurs after musical
           content has started, it is an unmistakable attention wrap-around restart.
        3. Multi-measure cycle & wrap-around detection:
           - Wrap-around to start: if measure i matches measure 0 after progress (i >= 2).
           - K-measure cycle detection for k in [1, 2, 3, 4]: detects [A, B, A, B] or [A, B, C, A, B, C].
           - Line-level consecutive repeats exceeding max_repeats.
        4. Guarantees balanced termination without syntax corruption.
        """
        if not raw_kern:
            return raw_kern

        text = (
            raw_kern.replace("<s>", " ")
            .replace("</s>", "")
            .replace("<t>", "\t")
            .replace("<b>", "\n")
        )
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            return raw_kern

        headers = []
        measures = []
        cur_m = []
        music_started = False
        consecutive_same_data = 0
        last_data_line = None
        consecutive_rest_measures = 0
        in_rest_measure = False

        for line in lines:
            is_header = line.startswith("**") or (line.startswith("*") and not music_started) or line.startswith("!")
            is_barline = line.startswith("=")

            # 1. Truncate at terminal barlines (e.g. '==', '=||', '=:|', '*-')
            if not is_header and any(t in line for t in ("==", "=||", "=:|", "*-")):
                if cur_m:
                    measures.append(cur_m)
                    cur_m = []
                measures.append([line])
                break

            # 2. Header re-occurrence detection (attention wrap-around)
            if music_started and line.startswith("*"):
                if any(line.startswith(h) for h in ("*clef", "*k[", "*M", "**kern", "*stria", "*tremolo", "*X8ba")):
                    break

            # 3. Token degeneration check (e.g. runaway repeated characters like 333333... or aaaaaa...)
            if not is_header and any(re.search(r'([0-9a-zA-Z])\1{5,}', tok) for tok in line.split("\t")):
                break

            if is_header:
                if (line in headers and any(h in line for h in ("*8va", "*X8va", "*15m", "*stria", "*tremolo"))) or len(headers) >= 12:
                    # Degenerative pre-music loop detected (e.g. runaway *8va cycle)
                    break
                headers.append(line)
                continue

            music_started = True

            if is_barline:
                if in_rest_measure:
                    consecutive_rest_measures += 1
                    if consecutive_rest_measures >= 2:
                        break
                else:
                    consecutive_rest_measures = 0
                in_rest_measure = True
                consecutive_same_data = 0
                last_data_line = None
                if cur_m:
                    measures.append(cur_m)
                    cur_m = []
                cur_m.append(line)
                continue

            # Data line: check consecutive repeats within measure
            tokens = line.split("\t")
            is_rest = all("r" in t or t == "." for t in tokens)
            if not is_rest:
                in_rest_measure = False

            if line == last_data_line:
                consecutive_same_data += 1
                if consecutive_same_data >= max_repeats:
                    # Runaway repetition detected within a measure (e.g. 64ee[ or 8G 8g)
                    break
            else:
                consecutive_same_data = 1
                last_data_line = line

            cur_m.append(line)

        if cur_m:
            measures.append(cur_m)

        # 3. Intelligent cycle and wrap-around detection (exact and fuzzy Jaccard similarity)
        cutoff = len(measures)

        # Extract pitch signature sets for robust invariant comparison
        def _measure_sig_set(m):
            notes = set()
            for line in m:
                if line.startswith("=") or line.startswith("*") or line.startswith("!"):
                    continue
                for t in line.split("\t"):
                    letters = "".join(sorted(re.findall(r"[a-gA-Gr]", t.lower())))
                    if letters:
                        notes.add(letters)
            return frozenset(notes)

        def _jaccard(s1, s2):
            if not s1 or not s2:
                return 0.0
            return len(s1 & s2) / float(len(s1 | s2))

        sigs = [_measure_sig_set(m) for m in measures]

        # A. Wrap-around to start: if measure i (i >= 2) matches measure 0
        if len(sigs) >= 3 and len(sigs[0]) > 0:
            for i in range(2, len(sigs)):
                if _jaccard(sigs[i], sigs[0]) >= 0.55:
                    cutoff = min(cutoff, i)
                    break

        # B. Multi-measure cycle detection for k in [1, 2, 3, 4]
        for k in (1, 2, 3, 4):
            limit = min(cutoff, len(sigs))
            min_cycles = 3 if k == 1 else 2
            if limit >= min_cycles * k:
                for end_idx in range(min_cycles * k, limit + 1):
                    pat1 = sigs[end_idx - k : end_idx]
                    pat2 = sigs[end_idx - 2 * k : end_idx - k]
                    sims = [_jaccard(s1, s2) for s1, s2 in zip(pat1, pat2)]
                    avg_sim = sum(sims) / float(len(sims)) if sims else 0
                    if avg_sim >= 0.55 and all(len(s) > 0 for s in pat1):
                        cutoff = min(cutoff, end_idx - k)
                        break

        valid_measures = measures[:cutoff]
        res_lines = list(headers)
        for m in valid_measures:
            res_lines.extend(m)

        # Count active spines for balanced terminator
        num_spines = 1
        if headers:
            num_spines = max(1, len(headers[0].split("\t")))
        elif valid_measures and valid_measures[0]:
            num_spines = max(1, len(valid_measures[0][0].split("\t")))

        # Ensure valid Humdrum termination
        if not res_lines or not res_lines[-1].startswith("*-"):
            if not res_lines or not res_lines[-1].startswith("="):
                res_lines.append("\t".join(["="] * num_spines))
            res_lines.append("\t".join(["*-"] * num_spines))

        return "\n".join(res_lines)

    def normalize_humdrum(self, raw_kern: str) -> str:
        """
        Ensures Humdrum text has valid spines, headers (**kern), and balanced column counts.
        Tracks active spine splits (*^) and merges (*v), padding missing fields to prevent Verovio C++ crashes.
        """
        sanitized = self.sanitize_runaway_kern(raw_kern)
        text = (
            sanitized.strip()
            .replace("<s>", " ")
            .replace("</s>", "")
            .replace("<t>", "\t")
            .replace("<b>", "\n")
        )
        if not text:
            raise ValueError("Empty notation received")

        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            raise ValueError("Notation contains no valid lines")

        # Normalize mispredicted guitar/lute tablature clefs into standard bass clef
        cleaned_lines = []
        for line in lines:
            if "*clefTAB" in line:
                line = re.sub(r"\*clefTAB\d*", "*clefF4", line)
            if "*stria" in line:
                continue
            cleaned_lines.append(line)
        lines = cleaned_lines

        # Strip trailing dangling spine manipulation lines before end that lack following notes/events
        while lines and all(t.startswith("*") and not t.startswith("*-") for t in lines[-1].split("\t")):
            lines.pop()

        if not lines:
            raise ValueError("Notation contains no valid data lines")

        # Check if header exists
        if not any(line.startswith("**") for line in lines):
            first_cols = max(1, len(lines[0].split("\t")))
            header = "\t".join(["**kern"] * first_cols)
            lines = [header] + lines

        active_spines = len(lines[0].split("\t"))
        fixed_lines = [lines[0]]

        for line in lines[1:]:
            tokens = line.split("\t")

            # Check for spine manipulation lines
            has_spine_op = any(t in ("*^", "*v") for t in tokens)
            if has_spine_op:
                if all(t.startswith("*") for t in tokens):
                    # Spine manipulation line itself MUST have active_spines columns
                    if len(tokens) < active_spines:
                        tokens = tokens + ["*"] * (active_spines - len(tokens))
                    elif len(tokens) > active_spines:
                        tokens = tokens[:active_spines]

                    # Validate *v: merging only makes sense if at least two *v appear and active_spines >= 2
                    v_count = tokens.count("*v")
                    if v_count < 2 or active_spines < 2:
                        tokens = ["*" if t == "*v" else t for t in tokens]

                    next_count = 0
                    i = 0
                    while i < len(tokens):
                        if tokens[i] == "*^":
                            next_count += 2
                            i += 1
                        elif tokens[i] == "*v":
                            next_count += 1
                            while i < len(tokens) and tokens[i] == "*v":
                                i += 1
                        else:
                            next_count += 1
                            i += 1
                    active_spines = max(1, next_count)
                    fixed_lines.append("\t".join(tokens))
                    continue
                else:
                    # Strip misplaced spine operators from regular data lines
                    tokens = [t if t not in ("*^", "*v") else "." for t in tokens]

            # Ensure data lines, barlines, interpretations match active spine count
            if any(t.startswith("=") for t in tokens):
                bar_tok = next(t for t in tokens if t.startswith("="))
                tokens = [bar_tok] * active_spines
            elif len(tokens) < active_spines:
                pad_val = "*-" if tokens[0].startswith("*-") else "."
                tokens = tokens + [pad_val] * (active_spines - len(tokens))
            elif len(tokens) > active_spines:
                tokens = tokens[:active_spines]

            fixed_lines.append("\t".join(tokens))

        # Remove trailing spine ops before terminator
        while fixed_lines and all(t.startswith("*") and not t.startswith("*-") for t in fixed_lines[-1].split("\t")):
            fixed_lines.pop()
            if fixed_lines:
                active_spines = len(fixed_lines[-1].split("\t"))

        # Check terminating *-
        if not fixed_lines or not fixed_lines[-1].startswith("*-"):
            terminator = "\t".join(["*-"] * active_spines)
            fixed_lines.append(terminator)

        return "\n".join(fixed_lines)

    def validate_with_verovio(self, kern_text: str) -> bool:
        """
        Uses Verovio toolkit to validate whether the Humdrum syntax is well-formed.
        Strictly guards against C++ exit(1) by verifying spine consistency in Python first.
        """
        if not kern_text or kern_text.startswith("% [OMR"):
            return False
        try:
            norm = self.normalize_humdrum(kern_text)
            if not self.check_spines_valid(norm):
                return False
            return bool(self._tk.loadData(norm))
        except Exception:
            return False

    def render_svg(self, kern_text: str, scale: int = 80) -> str:
        """
        Renders musical score to SVG using Verovio C++ engine.
        Uses tight excerpt options so the staff fills the viewport rather than an empty A4 page.
        """
        if not kern_text or kern_text.startswith("% [OMR"):
            return ""
        try:
            normalized = self.normalize_humdrum(kern_text)
            if not self.check_spines_valid(normalized):
                return ""
            options = {
                "pageWidth": 1200,
                "pageHeight": 220,
                "pageMarginTop": 10,
                "pageMarginBottom": 10,
                "pageMarginLeft": 15,
                "pageMarginRight": 15,
                "scale": scale,
                "adjustPageHeight": True,
                "breaks": "none",
                "header": "none",
                "footer": "none"
            }
            self._tk.setOptions(options)
            if self._tk.loadData(normalized):
                raw_svg = self._tk.renderToSVG(1)
                return self.flatten_svg(raw_svg)
        except Exception:
            return ""
        return ""

    @staticmethod
    def flatten_svg(svg_str: str) -> str:
        """
        Flattens nested <svg class="definition-scale" ...> generated by Verovio into a single
        valid SVG root element compatible with Qt SVG Tiny 1.2 and modern web viewports.
        """
        if not svg_str or "<svg" not in svg_str:
            return svg_str
        match = re.search(r'<svg\s+([^>]*class=["\']definition-scale["\'][^>]*)>', svg_str)
        if not match:
            return svg_str
        inner_attrs = match.group(1)
        vb_m = re.search(r'viewBox=["\']([^"\']+)["\']', inner_attrs)
        viewbox = vb_m.group(1) if vb_m else ""

        svg_str = re.sub(r'<svg\s+([^>]*class=["\']definition-scale["\'][^>]*)>', r'<g \1>', svg_str)
        svg_str = re.sub(r'</svg>(\s*</svg>\s*$)', r'</g>\1', svg_str)

        if viewbox and "viewBox" not in svg_str[:200]:
            svg_str = re.sub(r'<svg\s+', f'<svg viewBox="{viewbox}" ', svg_str, count=1)
        return svg_str

    def kern_to_abc(self, raw_kern: str, title: Optional[str] = None) -> str:
        """
        Converts Humdrum **kern to multi-voice ABC notation via MusicXML and xml2abc.
        Preserves all voices (V:1, V:2, grand staff, chords, dynamics).
        """
        normalized_kern = self.normalize_humdrum(raw_kern)
        
        # 1. Parse Humdrum with music21 and compile notation (suppress noisy internal stderr warnings)
        with contextlib.redirect_stderr(io.StringIO()):
            score = music21.converter.parse(normalized_kern, format="humdrum")
        try:
            score.makeNotation(inPlace=True)
        except Exception:
            pass
        
        # 2. Export to MusicXML (with automatic duration sanitization fallback)
        try:
            xml_tmp = Path(score.write("musicxml"))
        except Exception:
            # Complex/inexpressible duration fallback: quantize to nearest standard durations
            try:
                std_durations = [0.125, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0]
                for el in score.flatten().notesAndRests:
                    try:
                        ql = float(el.duration.quarterLength)
                        el.duration.quarterLength = min(std_durations, key=lambda d: abs(ql - d))
                    except Exception:
                        pass
                score.makeNotation(inPlace=True)
                xml_tmp = Path(score.write("musicxml"))
            except Exception:
                # Direct ABC synthesis fallback when MusicXML export cannot serialize notation
                return self._score_to_fallback_abc(score, title=title)

        try:
            xml_content = xml_tmp.read_text(encoding="utf-8")
        finally:
            xml_tmp.unlink(missing_ok=True)
            
        # 3. Convert MusicXML to ABC via Willem Vree's xml2abc
        # xml2abc's OptionParser parses sys.argv if not cleared, so we isolate it
        import sys
        orig_argv = sys.argv
        sys.argv = [sys.argv[0]]
        try:
            abc_text = convert_xml2abc(file_to_convert=xml_content, file_to_convert_is_txt=True)
        finally:
            sys.argv = orig_argv

        if not abc_text:
            # Fallback: synthesize valid minimal ABC header and rests from score
            t_line = f"T:{title}\n" if title else ""
            abc_text = f"X:1\n{t_line}M:4/4\nL:1/4\nK:C\nz4 |"
            
        # 4. Clean up boilerplate metadata headers if not needed
        cleaned_abc = self._clean_abc_output(abc_text, title=title)
        return cleaned_abc

    def _clean_abc_output(self, abc_text: str, title: Optional[str] = None) -> str:
        """
        Strips automatic 'Music21 Fragment' labels while preserving structural ABC headers.
        """
        lines = abc_text.strip().splitlines()
        result_lines = []
        
        for line in lines:
            # Strip default music21 titles
            if line.startswith("T:Music21 Fragment") or line.startswith("C:Music21"):
                continue
            if line.startswith("T:") and title:
                result_lines.append(f"T:{title}")
                continue
            result_lines.append(line)
            
        # If title specified and no T: was present, insert after X:1
        if title and not any(l.startswith("T:") for l in result_lines):
            for idx, l in enumerate(result_lines):
                if l.startswith("X:"):
                    result_lines.insert(idx + 1, f"T:{title}")
                    break
                    
        return "\n".join(result_lines)

    def _score_to_fallback_abc(self, score: Any, title: Optional[str] = None) -> str:
        """
        Synthesizes valid monophonic ABC notation directly from a music21 Score
        when MusicXML export cannot serialize complex inexpressible durations or ties.
        """
        notes_str = []
        try:
            for el in score.flatten().notesAndRests:
                if getattr(el, "isRest", False):
                    notes_str.append("z")
                elif getattr(el, "isChord", False):
                    chord_pitches = "".join(p.name.replace("-", "b") for p in el.pitches)
                    notes_str.append(f"[{chord_pitches}]")
                elif getattr(el, "isNote", False):
                    notes_str.append(el.name.replace("-", "b"))
                if len(notes_str) >= 32:
                    break
        except Exception:
            pass
        body = " ".join(notes_str) if notes_str else "z4"
        t_line = f"T:{title}\n" if title else ""
        return f"X:1\n{t_line}L:1/4\nM:none\nI:linebreak $\nK:C\nV:1 treble\nV:1\n{body} |"
