# Windows Portable Packaging (PyInstaller One-Folder)

This project uses a **Windows-first, portable one-folder** distribution shape.

## Build

From project root:

```powershell
scripts\build\build_portable_windows.bat
```

or:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\build\build_portable_windows.ps1
```

The build uses:

- `scripts/build/DATA586Portable.spec`
- PyInstaller **one-folder** mode (`COLLECT`)

## Output

Expected output folder:

```text
dist\DATA586Portable\
  DATA586TrainingLauncher.exe
  DATA586TrainingWorker.exe
  DATA586DataWorker.exe
  _internal\...
  model\
  model_specs\
  checkpoints\
  data\
  logs\
  scripts\assets\
  RUN_ME_FIRST.txt
```

## Runtime behavior

- No user-installed Python is required for packaged use.
- Runtime folders are external and visible next to the exe:
  - `model`
  - `model_specs`
  - `checkpoints`
  - `data`
  - `logs`
- Missing folders are auto-created on startup.

## Notes

- This phase intentionally does not include installer/signing/auto-updater.
- Distribution is portable one-folder by design (not one-file).
