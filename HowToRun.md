# How to Run

## Requirements

- Windows PowerShell
- MinGW GCC with OpenMP support
- Python 3

## Steps

1. Open PowerShell.

2. Go to the project folder:

   ```powershell
   cd "E:\Projects\C++\PDC\Project"
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

## Outputs

- Benchmark CSV files and logs: `results\`
- Performance plots: `results\plots\`
- Demo images: `demo\`
- Written report: `REPORT.md`
