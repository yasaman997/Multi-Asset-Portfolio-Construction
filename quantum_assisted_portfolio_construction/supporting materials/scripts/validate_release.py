from pathlib import Path
import ast
import json
import re
import subprocess
import zipfile
import nbformat

ROOT = Path(__file__).resolve().parents[1]
TOKENS = [
    "v" + "12",
    "v" + "10.2",
    "v" + "10_2",
    "v" + "9.1",
    "v" + "9_1",
    "correc" + "ted",
    "strong_" + "final",
    "standalone_" + "v3",
    "original_" + "copilot",
    "finance_aligned_" + "v",
    "defensible_" + "v",
]
FORBIDDEN = re.compile("|".join(re.escape(token) for token in TOKENS), re.IGNORECASE)
TEXT_EXTENSIONS = {".md", ".py", ".json", ".csv", ".txt", ".yml", ".yaml"}
REQUIRED_STEPS = [
    "step_01_financial_formulation",
    "step_02_quantum_formulation",
    "step_03_data_pipeline",
    "step_04_classical_baseline",
    "step_05_tunable_goals",
    "step_05q_hybrid_qaoa",
    "step_06_comparison",
    "step_07_validation",
    "step_08_presentation",
    "step_09_portfolio_copilot",
    "step_10_risk_adjusted_scoring",
]


def notebook_text(path: Path) -> str:
    notebook = nbformat.read(path, as_version=4)
    parts: list[str] = []
    for cell in notebook.cells:
        parts.append(cell.get("source", ""))
        for output in cell.get("outputs", []):
            value = output.get("text")
            if isinstance(value, list):
                parts.extend(value)
            elif isinstance(value, str):
                parts.append(value)
            data = output.get("data", {})
            for key in ("text/plain", "text/html"):
                value = data.get(key)
                if isinstance(value, list):
                    parts.extend(value)
                elif isinstance(value, str):
                    parts.append(value)
    return "\n".join(parts)


def html_text(path: Path) -> str:
    text = path.read_text(encoding="utf-8", errors="ignore")
    return re.sub(r"data:image/[^;]+;base64,[A-Za-z0-9+/=\s]+", "", text)


def office_text(path: Path) -> str:
    with zipfile.ZipFile(path) as archive:
        return "\n".join(
            archive.read(name).decode("utf-8", errors="ignore")
            for name in archive.namelist()
            if name.endswith(".xml")
        )


def pdf_text(path: Path) -> str:
    try:
        return subprocess.check_output(
            ["pdftotext", str(path), "-"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return ""


def visible_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".ipynb":
        return notebook_text(path)
    if suffix == ".html":
        return html_text(path)
    if suffix in TEXT_EXTENSIONS:
        return path.read_text(encoding="utf-8", errors="ignore")
    if suffix in {".docx", ".pptx"}:
        return office_text(path)
    if suffix == ".pdf":
        return pdf_text(path)
    return ""


def markdown_link_problems(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    problems: list[str] = []
    for target in re.findall(r"(?<!!)\[[^\]]+\]\(([^)]+)\)", text):
        target = target.strip().split("#", 1)[0]
        if not target or re.match(r"^(?:https?://|mailto:|#)", target):
            continue
        target = target.replace("%20", " ")
        if not (path.parent / target).resolve().exists():
            problems.append(f"broken link: {path.relative_to(ROOT)} -> {target}")
    return problems


def main() -> None:
    problems: list[str] = []
    for folder in REQUIRED_STEPS:
        if not (ROOT / "steps" / folder).is_dir():
            problems.append(f"missing step folder: {folder}")
    for path in ROOT.rglob("*"):
        if not path.is_file() or ".git" in path.parts:
            continue
        relative = path.relative_to(ROOT).as_posix()
        if FORBIDDEN.search(relative):
            problems.append(f"internal label in filename: {relative}")
        text = visible_text(path)
        if text and FORBIDDEN.search(text):
            problems.append(f"internal label in content: {relative}")
        if path.suffix.lower() == ".py":
            try:
                ast.parse(path.read_text(encoding="utf-8"))
            except SyntaxError as error:
                problems.append(f"python syntax error: {relative}: {error}")
        if path.suffix.lower() == ".ipynb":
            notebook = nbformat.read(path, as_version=4)
            errors = [
                output
                for cell in notebook.cells
                for output in cell.get("outputs", [])
                if output.get("output_type") == "error"
            ]
            if errors:
                problems.append(f"notebook error output: {relative}")
        if path.suffix.lower() == ".md":
            problems.extend(markdown_link_problems(path))

    step8_files = [p for p in (ROOT / "steps" / "step_08_presentation").iterdir() if p.is_file()]
    expected_step8 = ROOT / "steps" / "step_08_presentation" / "step_08_scientific_presentation_final.pptx"
    if step8_files != [expected_step8]:
        problems.append("Step 8 must contain only step_08_scientific_presentation_final.pptx")
    step10 = ROOT / "steps" / "step_10_risk_adjusted_scoring"
    panel_language = re.compile(r"\b" + "ju" + "dge" + r"\b|" + "ju" + "dge-friendly|\b" + "ju" + "dging" + r"\b", re.IGNORECASE)
    for path in step10.rglob("*"):
        if path.is_file() and panel_language.search(visible_text(path)):
            problems.append(f"panel-directed wording in Step 10: {path.relative_to(ROOT)}")

    caches = [
        path.relative_to(ROOT).as_posix()
        for path in ROOT.rglob("*")
        if path.is_dir() and path.name in {"__pycache__", ".pytest_cache"}
    ]
    problems.extend(f"cache directory present: {path}" for path in caches)
    if problems:
        raise SystemExit("\n".join(sorted(set(problems))))
    print("Release validation passed")


if __name__ == "__main__":
    main()
