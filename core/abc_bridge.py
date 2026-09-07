"""
ABC Bridge: Robust Music Notation Translation and Validation
Utilizes Verovio C++ engine for structural validation and rendering,
coupled with Willem Vree's xml2abc for pristine multi-voice ABC generation.
"""

from pathlib import Path
import re
from typing import Optional, Tuple
import music21
import verovio
from abc_xml_converter import convert_xml2abc


class ABCBridge:
    def __init__(self):
        self._tk = verovio.toolkit()

    def normalize_humdrum(self, raw_kern: str) -> str:
        """
        Ensures Humdrum text has valid spines and headers (**kern, *-).
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

        # Check if header exists
        if not any(line.startswith("**") for line in lines):
            first_cols = len(lines[0].split("\t"))
            header = "\t".join(["**kern"] * first_cols)
            lines = [header] + lines

        # Check terminating *-
        if not lines[-1].startswith("*-"):
            last_cols = len(lines[-1].split("\t"))
            terminator = "\t".join(["*-"] * last_cols)
            lines.append(terminator)
        else:
            if len(lines) >= 2:
                preceding_cols = len(lines[-2].split("\t"))
                current_term_cols = len(lines[-1].split("\t"))
                if current_term_cols != preceding_cols:
                    lines[-1] = "\t".join(["*-"] * preceding_cols)

        return "\n".join(lines)

    def validate_with_verovio(self, kern_text: str) -> bool:
        """
        Uses Verovio toolkit to validate whether the Humdrum syntax is well-formed.
        """
        try:
            return bool(self._tk.loadData(kern_text))
        except Exception:
            return False

    def render_svg(self, kern_text: str) -> str:
        """
        Renders musical score to SVG using Verovio C++ engine.
        Useful for visual verification without audio playback.
        """
        normalized = self.normalize_humdrum(kern_text)
        if self._tk.loadData(normalized):
            return self._tk.renderToSVG(1)
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
