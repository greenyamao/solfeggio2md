from pathlib import Path
import music21
import verovio
from abc_xml_converter import convert_xml2abc

sample_kern = """**kern\t**kern
*staff2\t*staff1
*clefF4\t*clefG2
*k[f#c#]\t*k[f#c#]
*M4/4\t*M4/4
=1\t=1
4D\t4d
4F#\t4f#
4A\t4a
4d\t4dd
=2\t=2
1D 1d\t1dd
==\t==
*-	*-"""

def test_bridge():
    print("1. Testing Verovio load & validation...")
    tk = verovio.toolkit()
    loaded = tk.loadData(sample_kern)
    print(f"   Verovio loaded Humdrum: {loaded}")
    
    print("2. Parsing Humdrum with music21 & exporting to MusicXML...")
    score = music21.converter.parse(sample_kern, format="humdrum")
    score.makeNotation(inPlace=True)
    
    # Export MusicXML
    xml_path = Path(score.write("musicxml"))
    xml_content = xml_path.read_text(encoding="utf-8")
    xml_path.unlink(missing_ok=True)
    print(f"   MusicXML generated! Length: {len(xml_content)} chars")
    
    print("3. Converting MusicXML to ABC via Willem Vree's xml2abc...")
    abc_output = convert_xml2abc(file_to_convert=xml_content, file_to_convert_is_txt=True)
    print("--- GENERATED ABC CODE ---")
    print(abc_output)
    print("--------------------------")
    return abc_output

if __name__ == "__main__":
    test_bridge()
