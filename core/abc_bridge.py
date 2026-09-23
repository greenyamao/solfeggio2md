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

    def normalize_humdrum(self, raw_kern: str) -> str:
        """
        Ensures Humdrum text has valid spines, headers (**kern), and balanced column counts.
        Tracks active spine splits (*^) and merges (*v), padding missing fields to prevent Verovio C++ crashes.
        """
        text = raw_kern.strip()
        if not text:
            raise ValueError("Empty notation received")

        # Replace SMT/Transcoda token separators if present
        text = (
            text.replace("<s>", " ")
            .replace("</s>", "")
            .replace("<t>", "\t")
            .replace("<b>", "\n")
        )

        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            raise ValueError("Notation contains no valid lines")

        # Strip trailing dangling spine manipulation lines before end that lack following notes/events
        while lines and all(t.startswith("*") and not t.startswith("*-") for t in lines[-1].split("\t")):
            lines.pop()

        if not lines:
            raise ValueError("Notation contains no valid data lines")

        # Check if header exists
        if not any(line.startswith("**") for line in lines):
            first_cols = len(lines[0].split("\t"))
            header = "\t".join(["**kern"] * first_cols)
            lines = [header] + lines

        # Dynamically track active spine count and pad/truncate inconsistent records
        active_spines = len(lines[0].split("\t"))
        fixed_lines = []

        for line in lines:
            tokens = line.split("\t")

            # Check for spine manipulation lines
            if all(t.startswith("*") for t in tokens):
                if any(t == "*^" or t == "*v" for t in tokens):
                    next_count = 0
                    i = 0
                    while i < len(tokens):
                        t = tokens[i]
                        if t == "*^":
                            next_count += 2
                            i += 1
                        elif t == "*v":
                            # Merge consecutive *v into 1 spine
                            next_count += 1
                            while i < len(tokens) and tokens[i] == "*v":
                                i += 1
                        else:
                            next_count += 1
                            i += 1
                    active_spines = max(1, next_count)
                    fixed_lines.append(line)
                    continue

            # Ensure data lines, barlines, interpretations match active spine count
            if len(tokens) < active_spines:
                pad_val = "*-" if tokens[0].startswith("*-") else "."
                tokens = tokens + [pad_val] * (active_spines - len(tokens))
            elif len(tokens) > active_spines:
                tokens = tokens[:active_spines]

            fixed_lines.append("\t".join(tokens))

        # Check terminating *-
        if not fixed_lines[-1].startswith("*-"):
            terminator = "\t".join(["*-"] * active_spines)
            fixed_lines.append(terminator)

        return "\n".join(fixed_lines)

    def validate_with_verovio(self, kern_text: str) -> bool:
        """
        Uses Verovio toolkit to validate whether the Humdrum syntax is well-formed.
        """
        if not kern_text or kern_text.startswith("% [OMR"):
            return False
        try:
            norm = self.normalize_humdrum(kern_text)
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
