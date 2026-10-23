"""Keep report mathematics structural and safe in the offline reader."""
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET
from zipfile import ZipFile

from lidar_project.visualization.report_math import extract_equations, MATHML


class ReportMathematics(unittest.TestCase):
    def report(self, directory, expression):
        path = Path(directory)/"equation.docx"
        source = f'''<w:document
            xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"
            xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math">
            <w:body><w:p><m:oMath>{expression}</m:oMath>
            <w:r><w:t>(1)</w:t></w:r></w:p></w:body></w:document>'''
        with ZipFile(path, "w") as archive:
            archive.writestr("word/document.xml", source)
        return path

    def test_preserves_nested_fraction_exponent_and_radical(self):
        # Flattening this expression changes both its value and its meaning.
        expression = '''<m:rad><m:deg/>
          <m:e><m:f><m:num><m:sSup>
            <m:e><m:sSub><m:e><m:r><m:t>σ</m:t></m:r></m:e>
              <m:sub><m:r><m:t>r</m:t></m:r></m:sub></m:sSub></m:e>
            <m:sup><m:r><m:t>2</m:t></m:r></m:sup></m:sSup></m:num>
            <m:den><m:r><m:t>2n</m:t></m:r></m:den>
          </m:f></m:e></m:rad>'''
        with tempfile.TemporaryDirectory() as directory:
            record = extract_equations(self.report(directory, expression))[0]
        namespace = {"m": MATHML}
        math = ET.fromstring(record["mathml"])
        self.assertEqual(len(math.findall(".//m:msqrt/m:mrow/m:mfrac", namespace)), 1)
        self.assertEqual(len(math.findall(".//m:msup/m:mrow/m:msub", namespace)), 1)
        self.assertIn("(σ_{r}^{2})/(2n)", record["plain"])

    def test_escapes_comparisons_and_rejects_unsupported_structure(self):
        with tempfile.TemporaryDirectory() as directory:
            record = extract_equations(self.report(directory,
                        '<m:r><m:t>x &lt; y &amp; z</m:t></m:r>'))[0]
            math = ET.fromstring(record["mathml"])
            self.assertIn("<", "".join(math.itertext()))
            with self.assertRaisesRegex(ValueError, "Unsupported Office Math"):
                extract_equations(self.report(directory, '<m:unexpected/>'))


if __name__ == "__main__":
    unittest.main()
