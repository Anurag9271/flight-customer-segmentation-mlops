# End-to-End Integration Report
## Flight Customer Segmentation — Day 10

## Objective
Verify that predictions are consistent across all three
layers of the pipeline:
1. Direct model prediction (pkl files loaded in Python)
2. Flask API running locally (python api.py)
3. Dockerized Flask API (docker run)

## Pipeline Architecture

 ## Test Results

### Prediction Consistency

| Customer | Direct Model | Flask API | Docker API | Match |
|----------|-------------|-----------|------------|-------|
| Champion | 2 (Champions) | 2 (Champions) | 2 (Champions) | ✓ |
| At-Risk | 3 (At-Risk) | 3 (At-Risk) | 3 (At-Risk) | ✓ |
| Occasional | 1 (Occasional) | 1 (Occasional) | 1 (Occasional) | ✓ |
| Loyal | 0 (Loyal) | 0 (Loyal) | 0 (Loyal) | ✓ |

All 4/4 predictions consistent across all 3 layers.

### Validation Tests

| Test Case | Expected | Received | Result |
|-----------|----------|----------|--------|
| Missing fields | 400 | 400 | ✓ PASS |
| Wrong data type | 400 | 400 | ✓ PASS |
| Out of range value | 400 | 400 | ✓ PASS |
| Valid request | 200 | 200 | ✓ PASS |

## Conclusion

The complete pipeline is verified end-to-end:
- Model pkl files load and predict correctly
- Flask API applies identical transforms as training
- Docker container produces same predictions as local API
- Validation correctly rejects all invalid inputs
- Error messages are clear and informative

## Files Involved

| File | Purpose |
|------|---------|
| outputs/models/scaler.pkl | Fitted StandardScaler |
| outputs/models/kmeans_model.pkl | Trained KMeans K=4 |
| api.py | Flask REST API |
| Dockerfile.flask | Docker build instructions |
| test_e2e.py | End-to-end test script |
| .github/workflows/ci.yml | CI automation |

## How to Run

```bash
# Layer 1 + 2: Local Flask
python api.py
python test_e2e.py

# Layer 3: Docker
docker run -d -p 5000:5000 \
  -v %cd%/outputs:/app/outputs flask-api
python test_e2e.py
``