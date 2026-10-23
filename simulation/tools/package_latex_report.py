"""Stage and package the editable LaTeX report with its demonstration assets."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
from zipfile import ZIP_DEFLATED, ZipFile

try:
    from tools.package_project import collect_files
except ModuleNotFoundError:  # Direct execution from the tools directory.
    from package_project import collect_files

PROJECT = Path(__file__).resolve().parents[1]
ARCHIVE_ROOT = "LiDAR_Midterm_LaTeX"
FIGURES = (
    "outputs/weather_report/weather_fov_los.png",
    "outputs/all_case_report/target_characteristics.png",
    "outputs/sensor_dependencies/sensor_dependencies.png",
    "outputs/sensor_dependencies/weather_measurements_compact.png",
    "outputs/navigation_validation/report_threshold_detection.png",
    "outputs/all_case_report/weather_trajectories.png",
    "outputs/all_case_report/weather_profiles.png",
    "outputs/all_case_report/weather_maps.png",
)
VIDEO_STEMS = (
    "environment_conditions", "sensor_state_best_vs_worst", "sampling_reflectance_factorial",
)
LATEX_FILES = ("main.tex", "README.md")
ORIGINAL_PDF = "outputs/reports/Midterm_Project_Report.pdf"


def _inside(path: Path, directory: Path) -> Path:
    try:
        path.resolve().relative_to(directory.resolve())
    except ValueError as exc:
        raise ValueError(f"An input or output links outside its project: {path.name}") from exc
    return path


def _assets(project: Path):
    for relative in FIGURES:
        yield project / relative, Path("figures") / Path(relative).name
    for stem in VIDEO_STEMS:
        for extension in ("mp4", "json"):
            relative = Path("simulalted_environments") / f"{stem}.{extension}"
            yield project / relative, relative


def stage_assets(project: Path = PROJECT) -> dict:
    """Copy report-owned assets without changing the editable TeX or bibliography."""
    project = Path(project).resolve()
    report = _inside(project / "latex_report", project)
    assets = list(_assets(project))
    missing = [str(file.relative_to(project)) for file, _ in assets if not file.is_file()]
    if missing:
        raise ValueError("Generate the required report assets first: " + ", ".join(missing))
    for source, relative in assets:
        _inside(source, project)
        destination = _inside(report / relative, report)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    visual = project / "index.html"
    if visual.is_file():
        _inside(visual, project)
        html = visual.read_text(encoding="utf-8").replace(
            f'href="{ORIGINAL_PDF}"', 'href="main.pdf"')
        (report / "visual_report.html").write_text(html, encoding="utf-8")
    return {"report_directory": str(report), "staged_assets": len(assets),
            "visual_report": (report / "visual_report.html").is_file()}


def _validate_tex(report: Path) -> None:
    """Reject external file references and require every directly included figure."""
    text = (report / "main.tex").read_text(encoding="utf-8")
    commands = r"\\(includegraphics\*?(?:\[[^\]]*\])?|input|include|bibliography|figureasset)\s*\{([^}]+)\}"
    for match in re.finditer(commands, text):
        for filename in match.group(2).split(","):
            filename = filename.strip()
            if match.group(1) == "figureasset":
                filename = "figures/" + filename
            # figureasset's graphic template is checked through each explicit
            # call below; TeX macro parameters are not literal filenames.
            if match.group(1).startswith("includegraphics") and re.fullmatch(r"figures/#\d", filename):
                continue
            if "\\" in filename or "#" in filename:
                raise ValueError("Use explicit portable file references in main.tex: " + filename)
            relative = PurePosixPath(filename)
            if relative.is_absolute() or ".." in relative.parts or re.match(r"^[A-Za-z]:", filename):
                raise ValueError("The LaTeX report contains an external file reference: " + filename)
            file = _inside(report / filename, report)
            candidates = [file] if file.suffix else [file.with_suffix(suffix)
                                                    for suffix in (".tex", ".bib", ".png", ".pdf", ".jpg")]
            if not any(candidate.is_file() for candidate in candidates):
                raise ValueError("The LaTeX report references a missing asset: " + filename)


def _pdf_pages(path: Path) -> int:
    """Return total PDF pages while enforcing the course's body-page limit.

    The course guidelines exclude the title page and reference section from
    the 5–10 page midterm limit. The report template gives each its own page,
    so those two pages are excluded here.
    """
    from pypdf import PdfReader
    pages = len(PdfReader(path).pages)
    body_pages = pages - 2
    if not 5 <= body_pages <= 10:
        raise ValueError(
            "The midterm report must contain 5–10 report-body pages excluding "
            f"the title page and references; compiled PDF has {pages} total pages "
            f"({body_pages} body pages)."
        )
    return pages


def _report_metadata(report: Path, pages: int) -> dict:
    text = (report / "main.tex").read_text(encoding="utf-8")
    return {"source": "main.tex", "pdf": "main.pdf", "pages": pages,
            "font_size_pt": 11 if re.search(r"\\documentclass\[[^\]]*11pt", text) else None,
            "line_spacing": 1.5 if r"\onehalfspacing" in text else None,
            "margin_in": 1 if re.search(r"margin\s*=\s*1in", text) else None,
            "columns": 1 if "twocolumn" not in text else 2,
            "body_pages": max(pages - 2, 0),
            "student_id_blank": bool(re.search(r"Student ID\s+\\rule\{", text))}


def _requirements_coverage() -> list[dict]:
    # This is an evidence index for the reader, not an automatic scientific-grade
    # claim. It points to the source and media which must be assessed together.
    return [
        {"id": "mathematical_model", "requirement": "Mathematical sensor model and physical principles",
         "report": "main.tex: Section 2 (sec:theory), Equations 1–14",
         "evidence": ["simulation/lidar_project/model/sensor.py", "simulation/lidar_project/model/materials.py"]},
        {"id": "virtual_sensor", "requirement": "Demonstrated simulation with targets and environmental conditions",
         "report": "main.tex: Sections 3–4 (sec:procedure and sec:results)",
         "evidence": ["simulation/configs/", "simulation/lidar_project/trajectory/engine.py",
                      "simulalted_environments/environment_conditions.mp4"]},
        {"id": "sensor_state", "requirement": "Measurements depend on sensor position/orientation and configuration",
         "report": "main.tex: Sections 2.2, 2.5, 3.1 and 4.2; fig:fov and fig:sensor",
         "evidence": ["figures/weather_fov_los.png", "figures/sensor_dependencies.png",
                      "simulalted_environments/sensor_state_best_vs_worst.mp4"]},
        {"id": "target_features", "requirement": "Measurements depend on target characteristics",
         "report": "main.tex: Sections 3.2, 4.1 and 4.2; fig:targets and fig:sensor",
         "evidence": ["figures/target_characteristics.png", "figures/sensor_dependencies.png",
                      "simulalted_environments/sampling_reflectance_factorial.mp4"]},
        {"id": "environment", "requirement": "Measurements depend on environmental conditions",
         "report": "main.tex: Sections 3.3, 4.3, 4.5 and 4.6; fig:weather, fig:navigation and fig:maps",
         "evidence": ["figures/weather_measurements_compact.png", "figures/weather_trajectories.png",
                      "figures/weather_profiles.png", "figures/weather_maps.png",
                      "simulalted_environments/environment_conditions.mp4"]},
        {"id": "effectiveness_analysis", "requirement": "Mathematical analysis and media explain model effectiveness",
         "report": "main.tex: Sections 2.5, 4 and 5; Equations 12–14 and fig:threshold",
         "evidence": ["figures/report_threshold_detection.png", "simulation/data/evaluation/",
                      "simulation/data/navigation_threshold/", "simulation/data/sampling_reflectance/"]},
        {"id": "editable_submission", "requirement": "Editable report, final PDF, code, figures and demonstration media",
         "report": "main.tex: title, abstract, introduction, body, conclusions and IEEE-style references",
         "evidence": ["main.tex", "main.pdf", "README.md", "figures/", "simulalted_environments/", "simulation/"]},
    ]


def _package_inputs(project: Path, report: Path):
    selected = [(report / name, name) for name in (*LATEX_FILES, "main.pdf")]
    # The source copy can regenerate report assets and compile the same LaTeX
    # document using the included staging tool; do not recursively copy builds.
    selected.extend((report / name, "simulation/latex_report/" + name)
                    for name in LATEX_FILES)
    bibliography = report / "references.bib"
    if bibliography.is_file():
        selected.append((bibliography, "references.bib"))
        selected.append((bibliography, "simulation/latex_report/references.bib"))
    selected.extend((report / relative, relative.as_posix()) for _, relative in _assets(project))
    if (report / "visual_report.html").is_file():
        selected.append((report / "visual_report.html", "visual_report.html"))
    # Keep evidence at data/, matching the simulation's expected input paths.
    selected.extend((file, "simulation/" + file.relative_to(project).as_posix())
                    for file in collect_files(project))
    # The legacy package selector intentionally omits most dotfiles. Git attributes
    # are a safe, explicit source file needed to preserve hashed CSV bytes.
    attributes = project / ".gitattributes"
    if attributes.is_file():
        selected.append((attributes, "simulation/.gitattributes"))
    seen = set()
    for file, relative in selected:
        if relative in seen:
            continue
        seen.add(relative)
        _inside(file, project)
        if not file.is_file():
            raise ValueError("A required package input is missing: " + relative)
        yield file, relative


def build_package(project: Path = PROJECT, output: Path | None = None) -> dict:
    project = Path(project).resolve()
    stage_assets(project)
    report = project / "latex_report"
    missing = [name for name in (*LATEX_FILES, "main.pdf") if not (report / name).is_file()]
    if missing:
        raise ValueError("Finish and compile the LaTeX report before packaging: " + ", ".join(missing))
    _validate_tex(report)
    pages = _pdf_pages(report / "main.pdf")
    original_html = project / "index.html"
    if original_html.is_file() and f'href="{ORIGINAL_PDF}"' in original_html.read_text(encoding="utf-8"):
        if not (project / ORIGINAL_PDF).is_file():
            raise ValueError("The simulation viewer references a missing PDF: " + ORIGINAL_PDF)
    inputs = list(_package_inputs(project, report))
    entries = [{"path": relative, "bytes": file.stat().st_size,
                "sha256": hashlib.sha256(file.read_bytes()).hexdigest()}
               for file, relative in inputs]
    manifest = {
        "format_version": 1,
        "title": "3D LiDAR midterm: editable LaTeX submission",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "primary_report": _report_metadata(report, pages),
        "requirements_coverage": _requirements_coverage(),
        "figures": [relative.as_posix() for _, relative in _assets(project)
                    if relative.parts[0] == "figures"],
        "videos": [f"simulalted_environments/{stem}.mp4" for stem in VIDEO_STEMS],
        "evidence": "simulation/data/",
        "simulation_source": "simulation/",
        "files": entries,
        "omitted": ["large raw navigation scan/history NPZ archives", "caches and local build logs",
                    "Git metadata and machine-specific configuration"],
        "limitations": "Internal virtual-model validation; coefficients are illustrative, not calibrated hardware.",
    }
    manifest_bytes = (json.dumps(manifest, indent=2) + "\n").encode("utf-8")
    output = Path(output or project / "outputs/submission/Midterm_Project_LaTeX.zip").resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".zip.tmp")
    try:
        with ZipFile(temporary, "w", ZIP_DEFLATED, compresslevel=6) as archive:
            for file, relative in inputs:
                archive.write(file, f"{ARCHIVE_ROOT}/{relative}")
            archive.writestr(f"{ARCHIVE_ROOT}/manifest.json", manifest_bytes)
        with ZipFile(temporary) as archive:
            for row in entries:
                stored = archive.read(f"{ARCHIVE_ROOT}/{row['path']}")
                if len(stored) != row["bytes"] or hashlib.sha256(stored).hexdigest() != row["sha256"]:
                    raise ValueError("Package verification failed: " + row["path"])
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    manifest_file = output.with_name("latex_deliverables_manifest.json")
    manifest_file.write_bytes(manifest_bytes)
    return {"zip": str(output), "manifest": str(manifest_file), "pages": pages,
            "files": len(entries), "bytes": output.stat().st_size}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Destination ZIP file.")
    parser.add_argument("--stage-only", action="store_true", help="Stage assets before compiling main.tex.")
    args = parser.parse_args(argv)
    if args.stage_only and args.output:
        parser.error("--output applies only when building the ZIP")
    try:
        result = stage_assets() if args.stage_only else build_package(output=args.output)
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(1, f"LaTeX package build failed: {exc}\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
