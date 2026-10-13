"""Launch-contract and artifact-integrity checks; no Windows execution is implied.

Run: python -m unittest discover -s tests -p test_bootstrap.py -v
An optional POWERSHELL_PARSER_DLL uses the .NET PowerShell AST parser. On Windows,
the installed powershell.exe/pwsh parser is used without executing bootstrap.ps1.
"""
from pathlib import Path
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
PS = (ROOT / 'bootstrap.ps1').read_text(encoding='utf-8-sig')


def requirements(name):
    lines = (ROOT / name).read_text(encoding='utf8').splitlines()
    return dict(line.strip().split('==', 1) for line in lines
                if line.strip() and not line.lstrip().startswith('#'))


class BootstrapContract(unittest.TestCase):
    def test_cmd_modes_and_exit_codes(self):
        cases = {'00_START.cmd': ('all', None), '01_GPU_REEXTRACT.cmd': ('extract', None),
                 '03_GPU_COMPARE.cmd': ('compare', 'all'), '10_BASELINE.cmd': ('all', 'baseline'),
                 '11_V1.cmd': ('all', 'v1'), '12_V2.cmd': ('all', 'v2'), '13_V3.cmd': ('all', 'v3')}
        for name, (mode, method) in cases.items():
            text = (ROOT / name).read_text(encoding='utf8')
            with self.subTest(name=name):
                self.assertIn('cd /d "%~dp0"', text)
                self.assertIn('"%~dp0bootstrap.ps1"', text)
                self.assertIn('-Mode ' + mode, text)
                if method:
                    self.assertIn('-Method ' + method, text)
                self.assertIn('set "TEAM_RUN_EXIT=%ERRORLEVEL%"', text)
                self.assertIn('exit /b %TEAM_RUN_EXIT%', text)
                self.assertLess(text.index('set "TEAM_RUN_EXIT=%ERRORLEVEL%"'), text.index('pause'))

    def test_scoped_installation_and_optional_cuda(self):
        self.assertIn("$venv = Join-Path $root '.venv'", PS)
        self.assertIn("if ($Mode -eq 'extract')", PS)
        self.assertIn("'--method',$Method", PS)
        self.assertIn("'--features','generated_features'", PS)
        self.assertNotIn('house_price_v10', PS)
        self.assertNotIn('SetExecutionPolicy', PS)
        self.assertNotIn('Set-ExecutionPolicy', PS)
        self.assertNotIn('pip config set', PS)
        self.assertNotIn('RunAs', PS)
        self.assertNotIn('Uninstall', PS)
        self.assertIn("Move-Item -LiteralPath $venv -Destination $previousVenv", PS)

    def test_requirements_match_expected_runtime(self):
        base = requirements('requirements.txt')
        self.assertEqual(base['numpy'], '2.2.6')
        self.assertEqual(base['scikit-learn'], '1.7.2')
        self.assertEqual(base['threadpoolctl'], '3.7.0')
        self.assertNotIn('torch', base)
        gpu = requirements('requirements-gpu.txt')
        self.assertEqual(gpu['transformers'], '4.56.2')
        self.assertEqual(requirements('requirements-torch.txt')['torch'], '2.7.1+cu128')
        for name in ('requirements.txt', 'requirements-gpu.txt', 'requirements-torch.txt'):
            for line in (ROOT / name).read_text().splitlines():
                if line.strip() and not line.lstrip().startswith('#'):
                    self.assertRegex(line.strip(), r'^[a-zA-Z0-9-]+==[0-9][a-zA-Z0-9.+-]*$')

    def test_verified_runtime_before_extract_execute(self):
        expected = 'a3a2f8ac71045d7bcf80366a7ec2cc5a1ba08112182a59e5df08a10242b8460a'
        self.assertIn("$runtimeSha256 = '" + expected + "'", PS)
        self.assertIn('https://github.com/astral-sh/python-build-standalone/releases/download/20250918/', PS)
        self.assertLess(PS.index('Portable Python archive checksum mismatch'), PS.index("Invoke-Checked $tar.Source"))
        self.assertLess(PS.index('Extracted Python executable checksum mismatch'), PS.index('$candidate = Probe-Python $portable', PS.index('Extracted Python executable checksum mismatch')))
        self.assertIn('Assert-Https $RuntimeMirrorUrl', PS)
        runtime = ROOT / 'runtime/python-3.11.13-windows-x86_64.tar.gz'
        if runtime.is_file():
            self.assertEqual(hashlib.sha256(runtime.read_bytes()).hexdigest(), expected)

    def test_offline_manifest_before_pip_and_online_fallback(self):
        self.assertIn("$File -eq 'requirements.txt' -and (Test-OfflineWheelhouse)", PS)
        self.assertIn("'--no-index','--find-links'", PS)
        self.assertIn('Offline wheel checksum mismatch', PS)
        self.assertIn('Offline wheelhouse contains an unverified file', PS)
        self.assertIn("https://pypi.org/simple", PS)
        manifest_path = ROOT / 'WHEELHOUSE_MANIFEST.json'
        directory = ROOT / 'wheelhouse'
        if manifest_path.is_file() and directory.is_dir():
            entries = json.loads(manifest_path.read_text(encoding='utf-8-sig'))['files']
            self.assertEqual(set(entries), {p.name for p in directory.iterdir() if p.is_file()})
            for name, item in entries.items():
                path = directory / name
                with self.subTest(wheel=name):
                    self.assertEqual(Path(name).name, name)
                    self.assertEqual(path.stat().st_size, item['bytes'])
                    self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), item['sha256'])

    def test_powershell_encoding_and_failure_record(self):
        self.assertTrue((ROOT / 'bootstrap.ps1').read_bytes().startswith(b'\xef\xbb\xbf'))
        self.assertIn('Start-Transcript -LiteralPath $log', PS)
        self.assertIn("$global:LASTEXITCODE = 9009", PS)
        self.assertIn("BOOTSTRAP_STATUS.json", PS)
        self.assertIn("BOOTSTRAP_ERROR.txt", PS)
        self.assertIn("exit $script:exitCode", PS)
        self.assertIn("if ($null -ne $lock)", PS)

    def test_powershell_syntax_without_execution(self):
        parser_dll = os.environ.get('POWERSHELL_PARSER_DLL')
        if parser_dll and shutil.which('dotnet'):
            result = subprocess.run(['dotnet', parser_dll, str(ROOT / 'bootstrap.ps1')],
                                    capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn('errors=0', result.stdout)
            return
        shell = shutil.which('pwsh') or shutil.which('powershell.exe')
        if not shell:
            self.skipTest('No PowerShell parser available; Windows execution remains unverified')
        with tempfile.TemporaryDirectory() as directory:
            check = Path(directory) / 'parse_only.ps1'
            check.write_text("param([string]$Source)\n$tokens=$null; $errors=$null\n"
                             "[System.Management.Automation.Language.Parser]::ParseFile($Source,[ref]$tokens,[ref]$errors) | Out-Null\n"
                             "if ($errors.Count -gt 0) { $errors | Format-List; exit 1 }; exit 0\n", encoding='utf-8-sig')
            result = subprocess.run([shell, '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', str(check),
                                     '-Source', str(ROOT / 'bootstrap.ps1')], capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == '__main__':
    unittest.main()
