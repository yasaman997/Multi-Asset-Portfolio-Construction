from pathlib import Path
import ast
import json
import re
import subprocess
import zipfile
import nbformat
import pandas as pd

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
]
FORBIDDEN = re.compile("|".join(re.escape(token) for token in TOKENS), re.IGNORECASE)
STEPS = [
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


def visible_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".ipynb":
        notebook = nbformat.read(path, as_version=4)
        values: list[str] = []
        for cell in notebook.cells:
            values.append(cell.get("source", ""))
            for output in cell.get("outputs", []):
                text = output.get("text")
                if isinstance(text, list):
                    values.extend(text)
                elif isinstance(text, str):
                    values.append(text)
                for key in ("text/plain", "text/html"):
                    text = output.get("data", {}).get(key)
                    if isinstance(text, list):
                        values.extend(text)
                    elif isinstance(text, str):
                        values.append(text)
        return "\n".join(values)
    if suffix == ".html":
        text = path.read_text(encoding="utf-8", errors="ignore")
        return re.sub(r"data:image/[^;]+;base64,[A-Za-z0-9+/=\s]+", "", text)
    if suffix in {".md", ".py", ".json", ".csv", ".txt", ".yml", ".yaml"}:
        return path.read_text(encoding="utf-8", errors="ignore")
    if suffix in {".docx", ".pptx"}:
        with zipfile.ZipFile(path) as archive:
            return "\n".join(
                archive.read(name).decode("utf-8", errors="ignore")
                for name in archive.namelist()
                if name.endswith(".xml")
            )
    if suffix == ".pdf":
        try:
            return subprocess.check_output(
                ["pdftotext", str(path), "-"],
                text=True,
                stderr=subprocess.DEVNULL,
            )
        except (FileNotFoundError, subprocess.CalledProcessError):
            return ""
    return ""


def test_required_steps_present() -> None:
    for folder in STEPS:
        assert (ROOT / "steps" / folder).is_dir()


def test_no_internal_iteration_labels() -> None:
    for path in ROOT.rglob("*"):
        if not path.is_file() or ".git" in path.parts:
            continue
        relative = path.relative_to(ROOT).as_posix()
        assert not FORBIDDEN.search(relative), relative
        text = visible_text(path)
        assert not FORBIDDEN.search(text), relative


def test_python_sources_compile() -> None:
    for path in ROOT.rglob("*.py"):
        ast.parse(path.read_text(encoding="utf-8"))


def test_notebooks_have_no_error_outputs() -> None:
    for path in ROOT.rglob("*.ipynb"):
        notebook = nbformat.read(path, as_version=4)
        errors = [
            output
            for cell in notebook.cells
            for output in cell.get("outputs", [])
            if output.get("output_type") == "error"
        ]
        assert not errors, path


def test_step_sources_are_concise() -> None:
    for path in (ROOT / "steps").rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        assert not re.search(r"(?m)^\s*#", source), path
        string_expressions = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ]
        assert not string_expressions, path
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                assert ast.get_docstring(node, clean=False) is None, path


def test_step10_status_is_not_overclaimed() -> None:
    status = json.loads((ROOT / "results" / "status_certificate.json").read_text())
    assert status["provisional_challenge_wide_leader"] == "Step 4 | Scenario aware"
    assert status["fully_certified_shortlist_leader"] == "Strict-warning classical"
    assert "PENDING" in status["universal_certified_winner_status"]
    assert status["supports_quantum_advantage_claim"] is False


def test_zero_breach_scorecard() -> None:
    frame = pd.read_csv(ROOT / "results" / "challenge_wide_zero_benchmark_scorecard.csv")
    assert (frame["hard_breaches"] == 0).all()
    assert frame.iloc[0]["portfolio"] == "Step 4 | Scenario aware"


def test_step8_contains_only_presentation() -> None:
    folder = ROOT / "steps" / "step_08_presentation"
    files = [path.name for path in folder.iterdir() if path.is_file()]
    assert files == ["step_08_scientific_presentation_final.pptx"]


def test_step10_has_no_panel_directed_wording() -> None:
    pattern = re.compile(r"\b" + "ju" + "dge" + r"\b|" + "ju" + "dge-friendly|\b" + "ju" + "dging" + r"\b", re.IGNORECASE)
    for path in (ROOT / "steps" / "step_10_risk_adjusted_scoring").rglob("*"):
        if path.is_file():
            assert not pattern.search(visible_text(path)), path


def test_final_qaoa_configuration_matches_executed_final_mode() -> None:
    path = ROOT / "steps" / "step_05q_hybrid_qaoa" / "results" / "final_qaoa_run_configuration.csv"
    frame = pd.read_csv(path, dtype=str)
    config = dict(zip(frame["parameter"], frame["value"]))
    expected = {
        "run_mode": "final",
        "selection_mode": "active_rebalance",
        "selection_objective": "incumbent_marginal_utility",
        "qubits": "11",
        "cardinality": "6",
        "feasible_supports": "462",
        "qaoa_reps": "2",
        "shots": "1024",
        "maxiter": "60",
        "independent_seeds": "20",
        "top_quantum_samples": "30",
        "maximum_subsets_to_refine": "15",
        "exact_candidates_to_refine": "16",
        "qaoa_aggregation": "0.25",
        "active_balance_strength": "1.0",
        "class_balance_strength": "0.0",
        "exact_enumeration_limit": "20000",
        "transpiler_optimization_level": "2",
    }
    for key, value in expected.items():
        assert config.get(key) == value, (key, config.get(key), value)


def test_active_rebalance_ranking_contains_all_462_states() -> None:
    path = ROOT / "steps" / "step_05q_hybrid_qaoa" / "results" / "active_rebalance_exact_qubo_ranking.csv"
    frame = pd.read_csv(path, dtype={"bitstring": str})
    assert len(frame) == 462
    assert frame["bitstring"].nunique() == 462
    assert frame["rank"].tolist() == list(range(1, 463))
    assert frame["bitstring"].map(len).eq(11).all()
    assert frame["bitstring"].map(lambda x: x.count("1") == 6).all()
    first = frame.iloc[0]
    assert first["bitstring"] == "11111100000"
    assert first["selected_tickers"] == "SGOV, BIL, SHY, QQQ, VUG, MTUM"


def test_ambiguous_target_recovery_summary_name_removed() -> None:
    folder = ROOT / "steps" / "step_05q_hybrid_qaoa" / "results"
    assert not (folder / "qubo_summary.csv").exists()
    assert (folder / "classical_target_recovery_diagnostic_summary.csv").is_file()


def test_csv_exports_have_no_legacy_multiindex_pseudoheader_rows() -> None:
    for path in ROOT.rglob("*.csv"):
        try:
            frame = pd.read_csv(path, dtype=str, keep_default_na=False)
        except Exception:
            continue
        if frame.empty:
            continue
        first = frame.iloc[0].astype(str).tolist()
        assert not any("Unnamed:" in value for value in first), path


def test_ci_workflow_is_cache_safe() -> None:
    workflow = (ROOT / ".github" / "workflows" / "validate.yml").read_text(encoding="utf-8")
    assert "PYTHONDONTWRITEBYTECODE" in workflow
    assert "-p no:cacheprovider" in workflow
    assert workflow.count("python scripts/validate_release.py") >= 2


def test_exported_qubo_coefficients_reproduce_462_state_ranking() -> None:
    import numpy as np
    folder = ROOT / "steps" / "step_05q_hybrid_qaoa" / "results"
    matrix_frame = pd.read_csv(folder / "active_rebalance_qubo_matrix.csv")
    tickers = matrix_frame.iloc[:, 0].tolist()
    Q = matrix_frame[tickers].to_numpy(dtype=float)
    linear_frame = pd.read_csv(folder / "active_rebalance_qubo_linear.csv")
    assert linear_frame["ticker"].tolist() == tickers
    q = linear_frame["linear_coefficient"].to_numpy(dtype=float)
    ranking = pd.read_csv(folder / "active_rebalance_exact_qubo_ranking.csv", dtype={"bitstring": str})
    energies = []
    for bits in ranking["bitstring"]:
        z = np.fromiter((int(c) for c in bits), dtype=float)
        energies.append(float(z @ Q @ z + q @ z))
    assert np.allclose(energies, ranking["energy"].to_numpy(dtype=float), atol=2e-10, rtol=0.0)


def test_qubo_coefficient_sensitivity_preserves_submitted_ground_state() -> None:
    path = ROOT / "steps" / "step_05q_hybrid_qaoa" / "results" / "active_rebalance_qubo_coefficient_sensitivity.csv"
    frame = pd.read_csv(path)
    assert len(frame) == 81
    assert frame["matches_base_support"].astype(bool).all()
    assert frame["best_bitstring"].astype(str).eq("11111100000").all()


def test_sampling_context_is_explicitly_noncomparative() -> None:
    path = ROOT / "steps" / "step_05q_hybrid_qaoa" / "results" / "sampling_baseline_context.csv"
    frame = pd.read_csv(path)
    assert len(frame) == 3
    note = " ".join(frame["interpretation"].astype(str)).lower()
    assert "not an algorithmically matched" in note
