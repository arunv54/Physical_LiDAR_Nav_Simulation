"""Check portable paths and report publication gates without running studies."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from zipfile import ZipFile

from tools import export_report as report


class ReportExport(unittest.TestCase):
    def test_managed_renderer_takes_precedence(self):
        with tempfile.TemporaryDirectory() as directory:
            bundled = Path(directory)/'managed-soffice'
            bundled.write_text('#!/bin/sh\n')
            bundled.chmod(0o755)
            with patch.object(report, 'bundled_soffice', return_value=bundled):
                self.assertEqual(report.resolve_soffice('/unrelated/desktop/soffice'), bundled.resolve())

    def test_missing_renderer_has_actionable_error(self):
        with patch.object(report, 'bundled_soffice', return_value=Path('/missing/bundled')):
            with patch.dict(os.environ, {}, clear=True):
                with self.assertRaisesRegex(RuntimeError, 'LIDAR_SOFFICE'):
                    report.resolve_soffice()
                with self.assertRaisesRegex(RuntimeError, 'not executable'):
                    report.resolve_soffice('/missing/renderer')

    def test_builder_reports_missing_inputs_from_another_working_folder(self):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory)/'saved-studies'
            result = subprocess.run(
                [sys.executable, str(report.PROJECT/'tools/build_report.py'),
                 '--data-dir', str(data), '--output-dir', str(Path(directory)/'report')],
                cwd=directory, capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
            self.assertIn(str(data/'navigation_threshold/summary.json'), result.stderr)
            self.assertFalse((Path(directory)/'report').exists())

    def test_saved_figure_mode_requires_target_and_fixed_weather_report_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory)/'saved-studies'
            figures = Path(directory)/'saved-figures'
            for relative in ('evaluation/geometry/summary.json', 'evaluation/response/summary.json',
                             'navigation_dependencies/summary.json', 'navigation_threshold/summary.json',
                             'sampling_reflectance/summary.json'):
                path = data/relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('{}')
            result = subprocess.run(
                [sys.executable, str(report.PROJECT/'tools/build_report.py'),
                 '--data-dir', str(data), '--figures-dir', str(figures),
                 '--output-dir', str(Path(directory)/'report'), '--use-saved-figures'],
                cwd=directory, capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
            self.assertIn('Render the required report figures first', result.stderr)
            for name in ('target_characteristics.png', 'weather_measurements.png',
                         'weather_trajectories.png', 'weather_profiles.png', 'weather_maps.png',
                         'weather_fov_los.png', 'report_threshold_detection.png', 'sensor_dependencies.png'):
                self.assertIn(name, result.stderr)
            self.assertFalse((Path(directory)/'report').exists())

    def test_saved_figure_flag_is_forwarded_only_to_word_builder(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(report.subprocess, 'run', return_value=
                              subprocess.CompletedProcess([], 0, '', '')) as run:
                with patch.object(report, 'docx_metadata', return_value={'equations': 19}):
                    report.export_report(directory, format='docx', use_saved_figures=True)
                self.assertIn('--use-saved-figures', run.call_args.args[0])
            with patch.object(report.subprocess, 'run') as run:
                with patch.object(report, 'docx_metadata', return_value={'equations': 19}), \
                        patch.object(report, 'resolve_soffice', return_value=Path('/renderer')), \
                        patch.object(report, 'convert_pdf', return_value={'pages': 10}):
                    report.export_report(directory, format='pdf', use_saved_figures=True)
                run.assert_not_called()

    def test_docx_metadata_requires_real_native_equations(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'report.docx'
            with ZipFile(path, 'w') as archive:
                archive.writestr('word/document.xml',
                                 '<w:document xmlns:w="http://schemas.openxmlformats.org/'
                                 'wordprocessingml/2006/main"/>')
            with self.assertRaisesRegex(RuntimeError, 'native editable'):
                report.docx_metadata(path)
            with ZipFile(path, 'w') as archive:
                archive.writestr('word/document.xml',
                                 '<w:document xmlns:w="http://schemas.openxmlformats.org/'
                                 'wordprocessingml/2006/main" xmlns:m="http://schemas.'
                                 'openxmlformats.org/officeDocument/2006/math">'
                                 '<m:oMath><m:r><m:t>r</m:t></m:r></m:oMath></w:document>')
            self.assertEqual(report.docx_metadata(path)['equations'], 1)

    def test_invalid_pdf_never_replaces_published_report(self):
        with tempfile.TemporaryDirectory() as directory:
            word = Path(directory)/'report.docx'
            word.write_bytes(b'input is not opened by the mocked renderer')
            published = word.with_suffix('.pdf')
            published.write_bytes(b'previous validated report')

            def fake_renderer(command, **kwargs):
                converted = Path(command[command.index('--outdir')+1])
                (converted/'report.pdf').write_bytes(b'invalid conversion')
                self.assertIn('-env:UserInstallation=', ' '.join(command))
                self.assertNotEqual(kwargs['env']['HOME'], str(Path.home()))
                return subprocess.CompletedProcess(command, 0, '', '')

            with patch.object(report.subprocess, 'run', side_effect=fake_renderer):
                with patch.object(report, 'validate_pdf', side_effect=RuntimeError('11 pages')):
                    with self.assertRaisesRegex(RuntimeError, '11 pages'):
                        report.convert_pdf(word, Path('/explicit/renderer'))
            self.assertEqual(published.read_bytes(), b'previous validated report')
            self.assertFalse(published.with_suffix('.pdf.tmp').exists())


if __name__ == '__main__':
    unittest.main()
