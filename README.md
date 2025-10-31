# Cheating Detection Service

This project combines face recognition, head pose estimation, and suspicious object detection into a single Flask-based service. It builds on the training artefacts present in the repository.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

The API listens on port 8000 by default.

### Health check

```bash
curl http://localhost:8000/health
```

### Run detection

Upload an image via multipart form-data:

```bash
curl -X POST http://localhost:8000/api/detect \
  -F "file=@/path/to/exam-frame.jpg"
```

Or with a base64-encoded JSON payload:

```bash
curl -X POST http://localhost:8000/api/detect \
  -H "Content-Type: application/json" \
  -d '{"image_base64": "<base64-image>"}'
```

### Add a face to the database

Register a new identity with one or more images:

```bash
curl -X POST http://localhost:8000/api/faces \
  -F "name=student_01" \
  -F "images=@/path/to/student_01_a.jpg" \
  -F "images=@/path/to/student_01_b.jpg"
```

Or using JSON with base64-encoded images:

```bash
curl -X POST http://localhost:8000/api/faces \
  -H "Content-Type: application/json" \
  -d '{
        "name": "student_01",
        "images": ["<base64-a>", "<base64-b>"]
      }'
```

The service stores the mean embedding similarly to the original `Face_Database_Training.ipynb` workflow.

## Output structure

Responses include recognised faces, detected suspicious objects, coarse head pose classification, and a high-level status (`clear` or `attention`).
