# Lift Height Analyzer

This repository automatically builds a **portable Windows application** using GitHub Actions.

The computer that runs the finished program does **not** need Python, pip, OpenCV, NumPy, Matplotlib, VS Code, or an internet connection.

## What happens automatically

Whenever the `main` branch is updated, GitHub starts a fresh Windows build machine and:

1. installs Python on the temporary GitHub runner;
2. installs the required Python packages;
3. packages `LiftHeightAnalyzer.py` with PyInstaller;
4. creates the portable `LiftHeightAnalyzer` folder;
5. creates `LiftHeightAnalyzer_Windows_x64.zip`;
6. uploads the finished ZIP as a GitHub Actions artifact.

The target laboratory/user computer only receives the finished ZIP.

## Get a ready Windows build

After pushing these files to GitHub:

1. Open the repository on GitHub.
2. Open **Actions**.
3. Select **Build Windows portable application**.
4. Open the latest successful run.
5. Under **Artifacts**, download **LiftHeightAnalyzer-Windows-x64**.
6. Extract the downloaded GitHub artifact.
7. Inside it is `LiftHeightAnalyzer_Windows_x64.zip`.
8. Send that ZIP to the end user.

The end user extracts it and double-clicks `LiftHeightAnalyzer.exe`.

## Run the build manually

The workflow also supports manual builds:

**Actions → Build Windows portable application → Run workflow**

No local Python installation is needed for this build.

## Permanent versioned release

Pushing a tag beginning with `v` also creates a GitHub Release and attaches the ready-to-send Windows ZIP.

Example:

```bash
git tag v1.0.0
git push origin v1.0.0
```

GitHub then creates a release containing:

`LiftHeightAnalyzer_Windows_x64.zip`

This is useful when you want a permanent version to send to other people.

## Repository structure

```text
LiftHeightAnalyzer/
├── .github/
│   └── workflows/
│       └── build-windows.yml
├── LiftHeightAnalyzer.py
├── requirements.txt
├── README.md
└── README_USER.txt
```

## End-user workflow

```text
LiftHeightAnalyzer_Windows_x64.zip
        ↓
extract
        ↓
LiftHeightAnalyzer/
├── LiftHeightAnalyzer.exe
├── _internal/
├── Experiments/
└── README_USER.txt
        ↓
double-click EXE
        ↓
select measurements folder
        ↓
Start analysis
        ↓
_analysis/
```

## Notes

The application is intentionally built in PyInstaller `--onedir` mode. The whole `LiftHeightAnalyzer` directory is portable, but `LiftHeightAnalyzer.exe` must remain beside its `_internal` directory.

GitHub-hosted Windows runners perform the build, so there is no requirement to configure a Windows/Python build environment on the development computer.
