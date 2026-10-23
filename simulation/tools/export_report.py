"""Build the editable report and export a real, page-checked PDF.

Examples:
    python tools/export_report.py --format all
    python tools/export_report.py --format docx --output-dir outputs/reports
    python tools/export_report.py --format pdf --soffice /path/to/soffice

The Word builder reads completed studies; it does not rerun simulations. PDF
export requires LibreOffice and never substitutes a placeholder PDF.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from xml.etree import ElementTree
from zipfile import BadZipFile, ZipFile

PROJECT = Path(__file__).resolve().parents[1]
REPORT_STEM = 'Midterm_Project_Report'


def bundled_soffice() -> Path:
    """The managed renderer is authoritative when this runtime is installed."""
    return (Path.home()/'.cache/codex-runtimes/codex-primary-runtime'
            '/dependencies/bin/override/soffice')


def resolve_soffice(configured: str | Path | None = None) -> Path:
    bundled = bundled_soffice()
    if bundled.is_file() and os.access(bundled, os.X_OK):
        return bundled.resolve()
    candidate = configured or os.environ.get('LIDAR_SOFFICE')
    if not candidate:
        raise RuntimeError(
            'PDF export requires LibreOffice. Set LIDAR_SOFFICE to its executable '
            'path or pass --soffice /path/to/soffice. The editable Word report '
            'can be built separately with --format docx.')
    renderer = Path(candidate).expanduser().resolve()
    if not renderer.is_file() or not os.access(renderer, os.X_OK):
        raise RuntimeError(f'LibreOffice executable is missing or not executable: {renderer}')
    return renderer


def docx_metadata(path: Path) -> dict:
    if not path.is_file():
        raise RuntimeError(f'Editable report not found: {path}. Build --format docx first.')
    try:
        with ZipFile(path) as archive:
            document = ElementTree.fromstring(archive.read('word/document.xml'))
    except (BadZipFile, KeyError, ElementTree.ParseError) as exc:
        raise RuntimeError(f'The editable report is not a valid Word document: {path}') from exc
    namespaces = {
        'm': 'http://schemas.openxmlformats.org/officeDocument/2006/math',
        'wp': 'http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing',
    }
    equations = len(document.findall('.//m:oMath', namespaces))
    figures = len(document.findall('.//wp:inline', namespaces))
    if not equations:
        raise RuntimeError('The report has no native editable Word equations.')
    return {'docx': str(path), 'equations': equations, 'figures': figures,
            'docx_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def validate_pdf(path: Path, min_pages: int = 5, max_pages: int = 10) -> int:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError('PDF validation requires pypdf. Install the project reports extra.') from exc
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError('LibreOffice did not produce a nonempty PDF.')
    reader = PdfReader(path)
    pages = len(reader.pages)
    if not min_pages <= pages <= max_pages:
        raise RuntimeError(f'The PDF has {pages} pages; the course report requires '
                           f'{min_pages}–{max_pages}. The invalid PDF was not published.')
    if not all((page.extract_text() or '').strip() for page in reader.pages):
        raise RuntimeError('The exported PDF contains a page without readable text.')
    return pages


def convert_pdf(docx: Path, renderer: Path, timeout_s: int = 180) -> dict:
    """Use isolated LibreOffice state and publish only a validated real PDF."""
    destination = docx.with_suffix('.pdf')
    with tempfile.TemporaryDirectory(prefix='lidar-report-export-') as temporary:
        temporary = Path(temporary)
        profile = temporary/'profile'
        home = temporary/'home'
        converted = temporary/'pdf'
        for directory in (profile, home, converted):
            directory.mkdir()
        environment = os.environ.copy()
        environment['HOME'] = str(home)
        environment['XDG_CONFIG_HOME'] = str(home/'config')
        environment['XDG_CACHE_HOME'] = str(home/'cache')
        command = [str(renderer), f'-env:UserInstallation={profile.as_uri()}',
                   '--headless', '--convert-to', 'pdf:writer_pdf_Export',
                   '--outdir', str(converted), str(docx.resolve())]
        try:
            result = subprocess.run(command, cwd=PROJECT, env=environment,
                                    capture_output=True, text=True, timeout=timeout_s)
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f'PDF export timed out after {timeout_s} seconds.') from exc
        if result.returncode:
            details = (result.stderr or result.stdout).strip()
            raise RuntimeError(f'LibreOffice PDF export failed ({result.returncode}): {details}')
        generated = converted/destination.name
        pages = validate_pdf(generated)
        staged = destination.with_suffix('.pdf.tmp')
        shutil.copyfile(generated, staged)
        staged.replace(destination)
    return {'pdf': str(destination), 'pages': pages, 'renderer': str(renderer),
            'pdf_sha256': hashlib.sha256(destination.read_bytes()).hexdigest()}


def export_report(output_dir: str | Path = PROJECT/'outputs/reports', *,
                  format: str = 'all', data_dir: str | Path | None = None,
                  figures_dir: str | Path | None = None,
                  soffice: str | Path | None = None, timeout_s: int = 180,
                  use_saved_figures: bool = False) -> dict:
    if format not in ('docx', 'pdf', 'all'):
        raise ValueError('format must be docx, pdf, or all')
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    docx = output_dir/(REPORT_STEM+'.docx')
    renderer = resolve_soffice(soffice) if format in ('pdf', 'all') else None
    if format in ('docx', 'all'):
        command = [sys.executable, str(PROJECT/'tools/build_report.py'),
                   '--output-dir', str(output_dir)]
        if data_dir is not None:
            command.extend(['--data-dir', str(Path(data_dir).expanduser().resolve())])
        if figures_dir is not None:
            command.extend(['--figures-dir', str(Path(figures_dir).expanduser().resolve())])
        if use_saved_figures:
            command.append('--use-saved-figures')
        environment = os.environ.copy()
        environment.setdefault('MPLBACKEND', 'Agg')
        result = subprocess.run(command, cwd=PROJECT, env=environment,
                                capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError('Word report build failed:\n'+(result.stderr or result.stdout).strip())
    metadata = docx_metadata(docx)
    if renderer is not None:
        metadata.update(convert_pdf(docx, renderer, timeout_s))
    metadata_path = output_dir/'report_metadata.json'
    metadata_path.write_text(json.dumps(metadata, indent=2)+'\n')
    return metadata


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--format', choices=('docx', 'pdf', 'all'), default='all')
    parser.add_argument('--output-dir', type=Path, default=PROJECT/'outputs/reports')
    parser.add_argument('--data-dir', type=Path, help='Saved study root (default: project data/)')
    parser.add_argument('--figures-dir', type=Path, help='Figure root (default: project outputs/)')
    parser.add_argument('--use-saved-figures', action='store_true',
                        help='Build Word using existing report PNGs without rerendering them')
    parser.add_argument('--soffice', type=Path, help='Explicit LibreOffice executable on other machines')
    parser.add_argument('--timeout', type=int, default=180, help='PDF conversion timeout in seconds')
    arguments = parser.parse_args(argv)
    try:
        result = export_report(arguments.output_dir, format=arguments.format,
                               data_dir=arguments.data_dir, figures_dir=arguments.figures_dir,
                               soffice=arguments.soffice, timeout_s=arguments.timeout,
                               use_saved_figures=arguments.use_saved_figures)
    except (RuntimeError, OSError, ValueError) as exc:
        parser.exit(1, f'Report export failed: {exc}\n')
    print(json.dumps(result))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
