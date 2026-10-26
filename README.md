# VeriShield

VeriShield is one student's face-crop XceptionNet study on FaceForensics++. The measured methods are DeepFakes, Face2Face, FaceSwap, and NeuralTextures, at raw quality and at H.264 QP 23 and QP 40, using the official 720/140/140 video split. The study adds leave-one-method-out, FGSM and PGD on Jas's own weights, one robustness improvement, and a small local demo so a panel can run that detector and read a decision.

The faculty guide has not signed the scope note yet, so this paragraph is the working cut.

## Run the local demo

Install the Python packages listed in `pyproject.toml` (including `python-multipart`), then start the API from the repository root:

```bash
uvicorn src.api.app:app --host 127.0.0.1 --port 8000
```

Set `CHECKPOINT_PATH` to a real checkpoint file before expecting a score. If that variable is unset, or the file is not there, `POST /api/detect` returns an error and does not invent a score. `GET /api/thresholds` and `POST /api/decision` still apply the frozen cuts in `configs/thresholds-val.json` and do not need the checkpoint or the dataset.

In another shell:

```bash
cd web
npm install
npm run dev
```

Open http://127.0.0.1:5173/ . The dev server proxies `/api` and `/health` to the API. The landing page is `/` and the upload flow is on that page and at `/check`.

`pytest tests/test_api.py` checks the threshold decision without a checkpoint and without dataset files.

The roadmap is `/cursor/stores/self/docs/final-year-roadmap.md`.

Data download is Phase 5, not this step. There is no GitHub remote yet.
