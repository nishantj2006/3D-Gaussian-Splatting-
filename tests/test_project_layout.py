"""Package migration contracts; no model loads or inference."""
import ast
import subprocess
import sys

from gsedit.runtime import PROJECT_ROOT, commands, module_command


def test_registry_modules_exist_and_parse():
    registry = commands()
    assert len(registry) == 82
    for name, module in registry.items():
        assert module.startswith('gsedit.')
        path = PROJECT_ROOT / (module.replace('.', '/') + '.py')
        assert path.is_file(), name
        ast.parse(path.read_text())
        assert not (PROJECT_ROOT / (name + '.py')).exists()


def test_module_command_preserves_alternate_python_and_cross_package_stage():
    assert module_command('generate_local_image.py', python='/env/bin/python') == [
        '/env/bin/python', '-m', 'gsedit.generation.generate_local_image']
    assert module_command('render_ply_preview.py') == [
        sys.executable, '-m', 'gsedit.rendering.render_ply_preview']


def test_repository_root_does_not_depend_on_calling_directory(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    assert (PROJECT_ROOT / 'scene' / 'gaussian_model.py').is_file()
    assert module_command('missing-script.py') == [
        sys.executable, str(PROJECT_ROOT / 'missing-script.py')]


def test_top_level_help_lists_commands_without_loading_models():
    result = subprocess.run([sys.executable, '-m', 'gsedit', '--list'],
                            cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    assert 'run_autonomous_edit' in result.stdout
    assert 'refine_shared_splats' in result.stdout
    assert 'assets:' in result.stdout


def test_no_old_first_party_imports_remain():
    names = set(commands())
    for directory in ('gsedit', 'tests', 'utils', 'scene'):
        for path in (PROJECT_ROOT / directory).rglob('*.py'):
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.ImportFrom) and not node.level:
                    assert node.module not in names, str(path)
                elif isinstance(node, ast.Import):
                    assert not names.intersection(alias.name for alias in node.names), str(path)
