# How to run the project

There are two ways to use this project:

- **The app** (part 1 and 2): a Windows window with buttons. Nothing to install.
  Use this for the demo.
- **From the source code** (part 3 and 4): build the C program yourself, run the
  tests and the experiments from the report. Needs a compiler and Python.

## 1. Start the app (no installation)

1. Download `PDC-Convolution.zip` from the GitHub release page:
   <https://github.com/ShayanAhmed12/Parallel-Image-Convolution/releases>
2. Right-click the zip > **Extract All**, and extract it anywhere (Desktop is fine).
   Do not run it from inside the zip.
3. Open the extracted `PDC-Convolution` folder and double-click
   **`PDC-Convolution.exe`**.
4. The first time, Windows may show "Windows protected your PC", because the exe is
   not signed. Click **More info**, then **Run anyway**.

The folder also has `pdc_conv.exe` (the C program that does all the work), the C
source code (`src\`), the final results (`results\`) and the documents. Keep the
files together: the app needs `pdc_conv.exe` next to it.

## 2. Using the app

The window has three tabs. At the bottom:

- **Command:** the exact `pdc_conv` command the app is running. You can type the
  same command in a terminal and get the same result. The app only runs the C
  program and shows what it writes; it does not filter or time anything itself.
- **Status bar:** what is happening, a progress bar, and this computer's CPU name
  and number of threads.

Press **Enter** to run the current tab. **Stop** (on any tab) ends a run.

### Filter tab: one image, one method

1. **Input:** choose *Generated test image* and a size (e.g. 2048), or
   *Image file* > **Browse...** for your own picture (JPG, PNG, BMP, TIFF, PPM).
2. **Filter:** Blur (choose the window size, 3 to 25), Sharpen, or Sobel edges.
3. **Method:** how the work is split between threads. The text under it explains
   the chosen method. For *Rows* you can also choose the **Schedule**
   (static, dynamic, guided). Choose the number of **Threads**.
4. Leave **Show thread map** ticked to see which thread computed each pixel.
5. Click **Apply**.

You see the original, the result and the thread map (each colour is one thread).
**Result** shows the parallel time, the sequential time, the speedup, and
**"Identical to sequential"** when every byte matches the sequential output.
**Save As...** (or File > Save result, Ctrl+S) saves the filtered image.

Try this: Rows + static gives solid bands of colour (one block of rows per thread).
Switch to dynamic: thin stripes, a different pattern every time.

### Benchmark tab: every method at several thread counts

1. Choose the image **Size**, the **Filter**, and tick the **Threads** to test
   (e.g. 1, 2, 4, 8).
2. **Repetitions:** each method runs this many times and the fastest counts.
3. Click **Run**.

Rows appear in the table one by one as they are measured, and the chart grows with
them. **Show:** switches the chart between Speedup, Efficiency and Time. The
dashed line is "ideal" (p threads = p times faster). The **Verified** column must say
`ok`; a red `FAIL` would mean a method gave a different image.

Sizes 8000 and 12000 take minutes. 2048 or 4096 is good for a demo.

### Batch tab: many small images

Choose the **Number of images** and the **Size of each**, the filter and threads,
then **Run**. It compares two ways of using the threads:

- **Threads share each image:** images one after another, each split by rows.
- **One image per thread:** each thread takes whole images.

The chart shows speedup or images per second.

### Menus

- **File > Open image...** (Ctrl+O): choose a picture for the Filter tab.
- **File > Save result...** (Ctrl+S): save the last filtered image.
- **Help > Open source folder:** opens the C source code.

## 3. Build and run from source (development PC)

### One-time setup

1. **C compiler:** WinLibs GCC, 64-bit (x86_64, UCRT), from <https://winlibs.com>.
   Extract it so that `C:\mingw64\bin\gcc.exe` exists. It must be 64-bit: the
   12000 x 12000 runs need about 3 GB of memory in one program.
2. **Python 3** from <https://python.org> (tick "Add python.exe to PATH").
3. In PowerShell, in the project folder:

```powershell
python -m pip install -r requirements.txt
```

### Build the C program

```powershell
.\scripts\build.ps1
```

This makes `pdc_conv.exe` in the project folder. If PowerShell refuses to run
scripts, use `powershell -ExecutionPolicy Bypass -File scripts\build.ps1`.
If your compiler is somewhere else: `.\scripts\build.ps1 -Compiler D:\path\to\gcc.exe`.

### Run the app from source

```powershell
pythonw gui\app.pyw
```

### Run the tests

```powershell
python -m unittest discover -s tests -v
```

About 40 tests, under a minute. They check that every method gives exactly the
same bytes as the sequential one on awkward image sizes, and that the app's tabs
work.

### Run all the experiments from the report

```powershell
python scripts\run_all.py
```

About 20 to 25 minutes on a 4-core PC. **Close other programs first**: timings need
the whole CPU. It writes `results\*.csv` and `*.log`, the graphs in
`results\plots\` and the example images in `demo\`.

`python scripts\run_all.py --quick` does the same steps on tiny images in seconds,
to check that everything works. **It overwrites `results\` and `demo\`**, so run it
only in a copy of the project, or restore them afterwards with
`git checkout -- results demo`.

To redraw only the graphs from the existing CSV files:

```powershell
python scripts\plot_results.py results
```

### Make the zip for another computer

```powershell
.\scripts\package.ps1
```

This makes `dist\PDC-Convolution.zip` (about 50 MB). The first time it creates a
small private Python environment in `build\venv` and downloads the three packages
from `requirements.txt` into it, so it needs internet once.

## 4. The command line

`pdc_conv.exe` can be used directly from PowerShell in the project folder. Some
useful examples:

```powershell
.\pdc_conv.exe gen --size 2048 --out test.ppm
```

Makes a 2048 x 2048 test image.

```powershell
.\pdc_conv.exe filter --in test.ppm --kernel blur --blur-size 9 --variant omp_rows --threads 4 --verify --out blurred.ppm
```

Blurs it with 4 threads, also runs the sequential version, and prints both times,
the speedup and whether the outputs are identical.

```powershell
.\pdc_conv.exe filter --in test.ppm --kernel sharpen --variant omp_rows --schedule dynamic --threads 4 --threadmap map.ppm
```

Also writes `map.ppm`, the thread map.

```powershell
.\pdc_conv.exe bench --size 4096 --kernel blur --threads 1,2,4,8 --reps 3 --csv bench.csv
```

Every method at 1, 2, 4 and 8 threads, as a table and a CSV file.

```powershell
.\pdc_conv.exe batch --n 500 --bsize 512 --threads 1,2,4,8
```

The many-small-images experiment.

Run `.\pdc_conv.exe` with no arguments to see every option. Method names:
`seq`, `omp_rows`, `omp_tiled`, `seq_sep`, `omp_sep` (the last two are blur only).
Exit codes: 0 = ok, 1 = error, 2 = an output differed from the sequential one.

PPM is a very simple image format. To view a `.ppm` file, open it in the app
(Filter tab > Image file), or in GIMP or IrfanView.

## 5. If something goes wrong

| Problem | What to do |
|---|---|
| "Windows protected your PC" | More info > Run anyway (the exe is not signed). |
| The app says it cannot start `pdc_conv.exe` | Keep `pdc_conv.exe` in the same folder as the app. Antivirus may have blocked it: allow it, or extract the zip again. |
| A benchmark takes very long | Use a smaller size (2048 or 4096), fewer thread counts, or fewer repetitions. Stop ends it at once. |
| Speedups are lower than in the report | Close other programs; plug in a laptop's charger (battery mode slows the CPU). Different CPUs give different numbers. |
| `build.ps1`: "Compiler not found" | Install WinLibs to `C:\mingw64`, or pass `-Compiler`. |
| `build.ps1`: running scripts is disabled | `powershell -ExecutionPolicy Bypass -File scripts\build.ps1` |
| `python` is not recognised | Reinstall Python with "Add python.exe to PATH" ticked. |
| "out of memory" on a very big image | Use a smaller size; 12000 x 12000 blur needs about 3 GB free. |
