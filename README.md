# Hand Sign Classifier

Small Python project to train and run a simple hand sign classifier.

This repository contains scripts to prepare data, train a model, and run inference on images or webcam input. It is organized for clarity and quick experimentation.

## Repository structure

- `app.py` - (likely) application entrypoint to run inference or serve the model.
- `data_input.py` - utilities for loading or preprocessing the dataset.
- `train_model.py` - training script that builds and trains the model.
- `utils.py` - shared helper functions used by the project.
- `dataset.csv` - CSV file listing dataset images and labels.
- `requirements.txt` - Python dependencies for the project.

> Note: The README assumes these files are the primary scripts. If your project uses different entrypoints, update the sections below accordingly.

## Requirements

- Python 3.8+ (3.10 recommended)
- See `requirements.txt` for exact package versions. Create and activate a virtual environment before installing packages.

Example (Windows PowerShell):

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1; pip install -r requirements.txt
```

If you get an execution policy error when activating the venv, run PowerShell as Administrator and allow script execution for the current user:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

## Quick start — Training

1. Prepare your dataset. Ensure `dataset.csv` contains one row per sample with at least `image_path` and `label` columns. Update `data_input.py` if your CSV schema differs.
2. Train the model:

```powershell
python train_model.py
```

The training script should save the trained model (check `train_model.py` for the output path). If you need to change hyperparameters (epochs, batch size, learning rate), edit `train_model.py` or expose CLI flags.

## Quick start — Inference / App

Run the application script to perform inference or start a small demo (webcam or local images):

```powershell
python app.py
```

Inspect `app.py` to see supported CLI arguments or how input images are provided.

## Dataset

This repository includes `dataset.csv` which lists your training images and labels. If you need to create a new CSV, the simplest format is:

```
image_path,label
data/0_a.png,A
data/1_b.png,B
```

Place image files under a `data/` directory (or edit `data_input.py` to match your layout).

## Development notes

- Add CLI flags to `train_model.py` for configurable training runs (epochs, batch size, model architecture).
- Add unit tests for `data_input.py` and `utils.py` to validate preprocessing steps.

## Contributing

1. Fork the repository.
2. Create a feature branch: `git checkout -b feature/your-feature`.
3. Commit your changes and push to your fork.
4. Open a Pull Request describing your changes.

## Pushing to GitHub (local steps)

```powershell
git add README.md; git commit -m "Add README"; git push origin master
```

Replace `master` with your main branch name if different (for example `main`).

## License

Add a LICENSE file to clarify the project's license (MIT is a popular choice for small projects).

---

If you want, I can tailor this README with exact CLI examples using the actual flags from `train_model.py` and `app.py` — tell me if you'd like that and I'll inspect those files and update the README.
