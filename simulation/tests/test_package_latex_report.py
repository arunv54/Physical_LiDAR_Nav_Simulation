"""Portable LaTeX assets, archive checksums and course page-limit gates."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from zipfile import ZipFile

from pypdf import PdfWriter

from tools import package_latex_report as package


class LatexPackage(unittest.TestCase):
    def make_project(self, project, pages=7):
        report = project / "latex_report"
        report.mkdir()
        (report / "main.tex").write_text(
            r"\documentclass[11pt,letterpaper]{article}\usepackage[margin=1in]{geometry}"
            r"\onehalfspacing\begin{document}"
            r"Author \rule{2in}{0.4pt} Student ID \rule{2in}{0.4pt}"
            r"\includegraphics{figures/weather_fov_los.png}\bibliography{references}"
            r"\end{document}")
        (report / "README.md").write_text("Compile main.tex; the demonstrations are local files.\n")
        (report / "references.bib").write_text("@misc{course,title={Course handout}}\n")
        writer = PdfWriter()
        for _ in range(pages):
            writer.add_blank_page(width=612, height=792)
        with (report / "main.pdf").open("wb") as stream:
            writer.write(stream)
        original = project / package.ORIGINAL_PDF
        original.parent.mkdir(parents=True)
        original.write_bytes((report / "main.pdf").read_bytes())
        for source, _ in package._assets(project):
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_bytes(b"fixture asset: " + source.name.encode())
        (project / "index.html").write_text(
            '<a href="outputs/reports/Midterm_Project_Report.pdf">PDF</a>')
        (project / "README.md").write_text("Original project documentation\n")
        (project / ".gitattributes").write_text("*.csv whitespace=cr-at-eol\n")
        for relative in ("lidar_project/model.py", "data/sampling_reflectance/runs.csv",
                         "data/sampling_reflectance/raw_scans.npz", "data/navigation_threshold/trials.npz",
                         "build/local.log", ".env"):
            path = project / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"fixture")
        return report

    def test_stage_keeps_editable_sources_and_rewrites_only_report_link(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            report = self.make_project(project)
            original = (report / "main.tex").read_bytes()
            result = package.stage_assets(project)
            self.assertEqual(result["staged_assets"], 14)
            self.assertEqual((report / "main.tex").read_bytes(), original)
            self.assertIn('href="main.pdf"', (report / "visual_report.html").read_text())
            for source, relative in package._assets(project):
                self.assertEqual(source.read_bytes(), (report / relative).read_bytes())

    def test_archive_has_complete_media_working_evidence_paths_and_verified_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            self.make_project(project)
            result = package.build_package(project)
            prefix = package.ARCHIVE_ROOT + "/"
            with ZipFile(result["zip"]) as archive:
                names = set(archive.namelist())
                manifest = json.loads(archive.read(prefix + "manifest.json"))
                self.assertEqual(manifest["primary_report"]["pages"], 7)
                self.assertEqual(manifest["primary_report"]["body_pages"], 5)
                self.assertEqual(manifest["primary_report"]["font_size_pt"], 11)
                self.assertEqual(manifest["primary_report"]["line_spacing"], 1.5)
                self.assertEqual(manifest["primary_report"]["margin_in"], 1)
                self.assertTrue(manifest["primary_report"]["student_id_blank"])
                requirements = {row["id"] for row in manifest["requirements_coverage"]}
                self.assertEqual(requirements, {"mathematical_model", "virtual_sensor", "sensor_state",
                                                "target_features", "environment", "effectiveness_analysis",
                                                "editable_submission"})
                self.assertEqual(len(manifest["figures"]), 8)
                self.assertEqual(len(manifest["videos"]), 3)
                self.assertIn(prefix + "simulation/data/sampling_reflectance/runs.csv", names)
                self.assertIn(prefix + "simulation/data/navigation_threshold/trials.npz", names)
                self.assertIn(prefix + "simulation/.gitattributes", names)
                self.assertNotIn(prefix + "simulation/data/sampling_reflectance/raw_scans.npz", names)
                self.assertFalse(any("local.log" in name or name.endswith("/.env") for name in names))
                self.assertIn('href="main.pdf"', archive.read(prefix + "visual_report.html").decode())
                self.assertIn(prefix + "main.pdf", names)
                self.assertIn(prefix + "simulation/" + package.ORIGINAL_PDF, names)
                for row in manifest["files"]:
                    self.assertNotIn(directory, row["path"])
                    self.assertEqual(hashlib.sha256(archive.read(prefix + row["path"])).hexdigest(),
                                     row["sha256"])

    def test_missing_video_manifest_blocks_package(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            self.make_project(project)
            (project / "simulalted_environments/environment_conditions.json").unlink()
            with self.assertRaisesRegex(ValueError, "environment_conditions.json"):
                package.build_package(project)

    def test_course_page_limit_and_missing_tex_asset_block_package(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            report = self.make_project(project, pages=13)
            with self.assertRaisesRegex(ValueError, "5–10 report-body pages"):
                package.build_package(project)
            (report / "main.tex").write_text(r"\includegraphics{figures/missing.png}")
            with self.assertRaisesRegex(ValueError, "missing asset"):
                package.build_package(project)

    def test_external_tex_and_symlinked_assets_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryDirectory() as outside:
            project = Path(directory)
            report = self.make_project(project)
            (report / "main.tex").write_text(r"\includegraphics{/tmp/private-image.png}")
            with self.assertRaisesRegex(ValueError, "external file reference"):
                package.build_package(project)
            source = project / package.FIGURES[0]
            source.unlink()
            external = Path(outside) / "private.png"
            external.write_bytes(b"private")
            source.symlink_to(external)
            with self.assertRaisesRegex(ValueError, "outside its project"):
                package.stage_assets(project)

    def test_figureasset_macro_checks_each_called_filename(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            report = self.make_project(project)
            (report / "main.tex").write_text(
                r"\newcommand{\figureasset}[4]{\includegraphics{figures/#1}}"
                r"\figureasset{weather_fov_los.png}{3in}{Caption}{fig:fov}")
            package.build_package(project)
            with (report / "main.tex").open("a") as stream:
                stream.write(r"\figureasset{missing.png}{3in}{Caption}{fig:missing}")
            with self.assertRaisesRegex(ValueError, "missing asset: figures/missing.png"):
                package.build_package(project)


if __name__ == "__main__":
    unittest.main()
