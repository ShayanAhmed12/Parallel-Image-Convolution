# How to Run

## Requirements

- Windows PowerShell
- 64-bit MinGW-w64 GCC with OpenMP support. The build script looks for
  `C:\mingw64\bin\gcc.exe` first (WinLibs GCC 16.2 x86_64 UCRT, from
  https://github.com/brechtsanders/winlibs_mingw, extracted to `C:\`), then `gcc`
  on PATH. Use `-Compiler <path>` to pick a different one.
- Python 3
- About 2.1 GB of free RAM for the 12000x12000 blur benchmark (close large
  programs such as browsers first, or results are distorted by paging)

## Steps

1. Open PowerShell.

2. Go to the project folder:

   ```powershell
   cd "D:\Projects\C++\PDC\Project"
   ```

3. Install the Python dependencies once:

   ```powershell
   python -m pip install -r requirements.txt
   ```

4. Run the complete project workflow:

   ```powershell
   .\scripts\run_windows.ps1
   ```

5. For a shorter local test:

   ```powershell
   .\scripts\run_windows.ps1 -Size 4096 -BatchN 300 -BatchSize 512 -Reps 1
   ```

To only build: `.\scripts\build_windows.ps1` (or
`.\scripts\build_windows.ps1 -Compiler C:\path\to\gcc.exe`).

## Outputs

- Benchmark CSV files and logs: `results\`
- Performance plots: `results\plots\`
- Demo images: `demo\`
- Written report: `REPORT.md`
