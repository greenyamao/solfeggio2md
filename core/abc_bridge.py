"""
ABC Bridge: Robust Music Notation Translation and Validation
Utilizes Verovio C++ engine for structural validation and rendering,
coupled with Willem Vree's xml2abc for pristine multi-voice ABC generation.
"""

from pathlib import Path
import logging
import re
from typing import Optional, Tuple
import music21
import verovio
from abc_xml_converter import convert_xml2abc

# Silence verbose music21 terminal warnings (e.g. unknown clef types, unterminated spines)
logging.getLogger("music21").setLevel(logging.ERROR)


class ABCBridge:
    def __init__(self):
        try:
            verovio.enableLog(verovio.LOG_OFF)
        except Exception:
            pass
        self._tk = verovio.toolkit()

        # Suppress third-party MIDI channel and syntax warning spam from music21
        import logging
        for logger_name in ("music21", "music21.humdrum", "music21.musicxml"):
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

    def normalize_humdrum(self, raw_kern: str) -> str:
        """
        Ensures Humdrum text has valid spines, headers (**kern), and balanced column counts.
        Tracks active spine splits (*^) and merges (*v), padding missing fields to prevent Verovio C++ crashes.
        """
        text = (
            raw_kern.strip()
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
            if len(tokens) < active_spines:
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
                return self._tk.renderToSVG(1)
        except Exception:
            return ""
        return ""

    def kern_to_abc(self, raw_kern: str, title: Optional[str] = None) -> str:
        """
        Converts Humdrum **kern to multi-voice ABC notation via MusicXML and xml2abc.
        Preserves all voices (V:1, V:2, grand staff, chords, dynamics).
        """
        normalized_kern = self.normalize_humdrum(raw_kern)
        
        # 1. Parse Humdrum with music21 and compile notation
        score = music21.converter.parse(normalized_kern, format="humdrum")
        score.makeNotation(inPlace=True)
        
        # 2. Export to MusicXML (music21's XML exporter is lossless and robust)
        xml_tmp = Path(score.write("musicxml"))
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
            raise ValueError("Failed to generate ABC notation from MusicXML")
            
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
