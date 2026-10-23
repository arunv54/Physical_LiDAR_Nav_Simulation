"""Read the report's editable equations as offline browser MathML.

This module translates the saved Word report; it does not rebuild the report or
evaluate a sensor model. Unsupported Office Math constructs raise an error rather
than silently flattening a fraction, exponent, or radical.
"""

from __future__ import annotations

from pathlib import Path
import re
import unicodedata
import xml.etree.ElementTree as ET
from zipfile import ZipFile


WORD = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
OFFICE_MATH = "http://schemas.openxmlformats.org/officeDocument/2006/math"
MATHML = "http://www.w3.org/1998/Math/MathML"
NS = {"w": WORD, "m": OFFICE_MATH}
ET.register_namespace("", MATHML)

EQUATION_CAPTIONS = (
    "Discrete azimuth and elevation sampling",
    "Beam direction in the sensor frame",
    "Firing time and sensor translation",
    "Continuous angular and range field of view",
    "Nearest opaque surface intersection",
    "Diffuse and specular surface response",
    "Returned power with range and atmospheric attenuation",
    "Pulse gate, merged power, and echo centroid",
    "Single-echo detection probability",
    "Conditional range variance and measured range",
    "Measured points in sensor and world coordinates",
    "Ground range sensitivity to height and elevation",
    "Power sensitivity to range, extinction, and incidence",
    "Conditional range error decomposition",
    "Discrete robot motion",
    "Observed and predicted detection fractions",
    "Pointwise Wilson interval for a detection fraction",
    "Detected-return range and standardized errors",
    "Driven path length and observed endpoint-cell fraction",
)

_FUNCTIONS = {"cos", "sin", "tan", "exp", "min", "max", "wrap", "ln", "RMSE"}
_SUPER = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺", "0123456789−+")
_SUPER_CHARS = set("⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺")


def _node(tag: str, text: str | None = None, **attributes: str) -> ET.Element:
    result = ET.Element(f"{{{MATHML}}}{tag}", attributes)
    result.text = text
    return result


def _tokens(text: str, normal: bool = False) -> list[ET.Element]:
    """Tokenize an Office Math run without losing literal symbols or spacing."""
    result = []
    index = 0
    while index < len(text):
        char = text[index]
        if char.isspace():
            stop = index + 1
            while stop < len(text) and text[stop].isspace():
                stop += 1
            result.append(_node("mspace", width=f"{.18 * (stop-index):.2f}em"))
            index = stop
            continue
        if char in _SUPER_CHARS and result:
            stop = index + 1
            while stop < len(text) and text[stop] in _SUPER_CHARS:
                stop += 1
            exponent = text[index:stop].translate(_SUPER)
            power = _node("msup")
            power.extend((result.pop(), _node("mn", exponent)))
            result.append(power)
            index = stop
            continue
        number = re.match(r"(?:\d+(?:\.\d*)?|\.\d+)", text[index:])
        if number:
            token = number.group()
            result.append(_node("mn", token))
            index += len(token)
            continue
        function = next((value for value in sorted(_FUNCTIONS, key=len, reverse=True)
                         if text.startswith(value, index)), None)
        if function:
            result.append(_node("mi", function, mathvariant="normal"))
            index += len(function)
            continue
        if char.isalpha():
            stop = index + 1
            while stop < len(text) and unicodedata.combining(text[stop]):
                stop += 1
            token = text[index:stop]
            attributes = {"mathvariant": "normal"} if normal else {}
            result.append(_node("mi", token, **attributes))
            index = stop
            continue
        if unicodedata.combining(char):
            # Combining marks normally attach to the preceding identifier above.
            result.append(_node("mtext", char))
        else:
            result.append(_node("mo", char))
        index += 1
    return result


def _local(element: ET.Element) -> str:
    return element.tag.rsplit("}", 1)[-1]


def _row(element: ET.Element) -> ET.Element:
    result = _node("mrow")
    for child in element:
        for converted in _convert(child):
            result.append(converted)
    return result


def _required(element: ET.Element, name: str) -> ET.Element:
    child = element.find(f"m:{name}", NS)
    if child is None:
        raise ValueError(f"Office Math {_local(element)} is missing {name}")
    return child


def _convert(element: ET.Element) -> list[ET.Element]:
    name = _local(element)
    if name in {"rPr", "ctrlPr"} or name.endswith("Pr"):
        return []
    if name == "r":
        normal = element.find("m:rPr/m:sty", NS)
        normal = normal is not None and normal.get(f"{{{OFFICE_MATH}}}val") == "p"
        return _tokens("".join(element.itertext()).strip("\n"), normal)
    if name in {"oMath", "e", "sub", "sup", "num", "den", "deg"}:
        return [_row(element)]
    if name in {"sSub", "sSup", "sSubSup"}:
        result = _node({"sSub": "msub", "sSup": "msup", "sSubSup": "msubsup"}[name])
        result.append(_row(_required(element, "e")))
        if name != "sSup":
            result.append(_row(_required(element, "sub")))
        if name != "sSub":
            result.append(_row(_required(element, "sup")))
        return [result]
    if name == "f":
        result = _node("mfrac")
        result.extend((_row(_required(element, "num")), _row(_required(element, "den"))))
        return [result]
    if name == "rad":
        degree = element.find("m:deg", NS)
        hidden = element.find("m:radPr/m:degHide", NS)
        is_square = (degree is None or not list(degree) or
                     (hidden is not None and hidden.get(f"{{{OFFICE_MATH}}}val") == "1"))
        result = _node("msqrt" if is_square else "mroot")
        result.append(_row(_required(element, "e")))
        if not is_square:
            result.append(_row(degree))
        return [result]
    raise ValueError(f"Unsupported Office Math element: {name}")


def _plain(element: ET.Element) -> str:
    """A structural fallback for search/accessibility; MathML is the display."""
    name = _local(element)
    if name in {"rPr", "ctrlPr"} or name.endswith("Pr"):
        return ""
    if name == "r":
        return "".join(element.findall("m:t", NS)[i].text or ""
                       for i in range(len(element.findall("m:t", NS))))
    if name == "f":
        return f"({_plain(_required(element, 'num'))})/({_plain(_required(element, 'den'))})"
    if name in {"sSub", "sSup", "sSubSup"}:
        result = _plain(_required(element, "e"))
        if name != "sSup":
            result += "_{" + _plain(_required(element, "sub")) + "}"
        if name != "sSub":
            result += "^{" + _plain(_required(element, "sup")) + "}"
        return result
    if name == "rad":
        degree = element.find("m:deg", NS)
        if degree is None or not list(degree):
            return "sqrt(" + _plain(_required(element, "e")) + ")"
        return "root(" + _plain(degree) + ", " + _plain(_required(element, "e")) + ")"
    return "".join(_plain(child) for child in element)


def extract_equations(docx_path: str | Path) -> list[dict]:
    """Return numbered MathML copied structurally from the saved report DOCX.

    Each record has ``number``, ``mathml``, ``plain``, and ``caption``. Rendering
    uses browser-native MathML, so no CDN, JavaScript mathematics library, or
    network connection is required.
    """
    with ZipFile(docx_path) as archive:
        document = ET.fromstring(archive.read("word/document.xml"))
    records = []
    for paragraph in document.findall(".//w:p", NS):
        equations = paragraph.findall("m:oMath", NS)
        if not equations:
            continue
        if len(equations) != 1:
            raise ValueError("Expected one numbered equation per report paragraph")
        label = "".join(paragraph.itertext())
        match = re.search(r"\((\d+)\)\s*$", label)
        if not match:
            raise ValueError("The saved report equation has no numeric label")
        number = int(match.group(1))
        expression = equations[0]
        plain = _plain(expression)
        math = _node("math", display="block")
        math.set("aria-label", plain)
        math.append(_row(expression))
        records.append({
            "number": number,
            "mathml": ET.tostring(math, encoding="unicode"),
            "plain": plain,
            "caption": (EQUATION_CAPTIONS[number-1]
                        if 0 < number <= len(EQUATION_CAPTIONS)
                        else f"Report equation {number}"),
        })
    numbers = [record["number"] for record in records]
    if numbers != list(range(1, len(records)+1)):
        raise ValueError("The saved report equations must be consecutive and ordered")
    return records
