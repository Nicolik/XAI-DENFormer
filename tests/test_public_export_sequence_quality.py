from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_export_module():
    root = Path(__file__).resolve().parents[1]
    module_path = root / "tools" / "export_public.py"
    spec = importlib.util.spec_from_file_location("export_public", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_sequence_quality_files_are_required_in_public_export():
    module = _load_export_module()
    required = set(module.REQUIRED_PUBLIC_FILES)
    assert "classifier/workflow/reports/make_sequence_quality_baseline.py" in required
    assert "classifier/workflow/reports/SEQUENCE_QUALITY_BASELINE.md" in required
    assert "tests/test_sequence_quality_baseline.py" in required


def test_sequence_quality_commands_are_required_in_public_templates():
    module = _load_export_module()
    readme_requirements = module.REQUIRED_PUBLIC_CONTENT["README.md"]
    pipeline_requirements = module.REQUIRED_PUBLIC_CONTENT["pipeline.sh"]
    assert "make_sequence_quality_baseline" in readme_requirements
    assert "SEQUENCE_QUALITY_BASELINE.md" in readme_requirements
    assert "make_sequence_quality_baseline" in pipeline_requirements
