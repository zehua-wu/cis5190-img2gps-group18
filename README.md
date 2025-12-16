# Img2GPS Submission (Project A)

This folder contains the required submission files for Project A (Img2GPS).  
The backend will run **inference only** using the provided pretrained weights.

## Contents
- `model.py`  
  Model implementation. Exposes `get_model()` and a `Model` class that can be instantiated without extra arguments.
- `preprocess.py`  
  Data loader that exposes `prepare_data(csv_path) -> (X, y)`.
- `model.pt`  
  Trained PyTorch `state_dict` used for inference.
- `requirements.txt`  
  Python dependencies for inference.

## I/O Contract (Backend)
### Preprocess
`preprocess.prepare_data(csv_path)` returns `(X, y)` where:
- `X`: a list of image paths (strings). These will be batched and passed to `Model.predict(batch)`.
- `y`: a list of raw GPS labels `[lat, lon]` in **degrees** (not normalized).  
  (Used for local debugging; backend will compute metrics from CSV labels.)

The CSV is expected to contain image path + GPS columns. Common supported column names include:
- image path: `image_path`, `filepath`, `file_name`, `image`, `path`
- latitude: `Latitude`, `latitude`, `lat`
- longitude: `Longitude`, `longitude`, `lon`

### Model
The backend will instantiate the model from `model.py` and load `model.pt` (if provided).  
At inference, it calls:
- `model.predict(batch)` if available, otherwise `model(batch)`.

The model output must be:
- a list/array of `[lat, lon]` in **degrees (raw)** for each input in the batch.

## Local Evaluation (Recommended)
We verified the submission using the provided local evaluator:

```bash
python eval_project_a.py \
  --model submission/model.py \
  --preprocess submission/preprocess.py \
  --weights submission/model.pt \
  --csv reference/metadata.csv \
  --batch-size 32

eval_project_a.py is provided by the course staff and is not included in this submission.