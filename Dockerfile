# Dockerfile
# ──────────────────────────────────────────────────────────
# Builds a reproducible environment for the flight
# customer segmentation project.
# 
# Build:  docker build -t flight-segmentation .
# Run:    docker run -v %cd%/data:/app/data 
#                   -v %cd%/outputs:/app/outputs 
#                   flight-segmentation
# ──────────────────────────────────────────────────────────

# ── Layer 1: Base image ───────────────────────────────────
# python:3.10-slim = Python 3.10 on minimal Linux
# "slim" means no unnecessary tools — smaller image size
FROM python:3.10-slim

# ── Layer 2: Set working directory ────────────────────────
# All commands after this run inside /app
# All COPY commands copy into /app
WORKDIR /app

# ── Layer 3: Copy requirements FIRST ──────────────────────
# We copy requirements.txt BEFORE copying code
# Reason: if only code changes, Docker reuses the
# pip install layer from cache — much faster rebuild
COPY requirements-docker.txt .

# ── Layer 4: Install dependencies ─────────────────────────
# --no-cache-dir = don't store pip cache inside image
# keeps image size smaller
RUN pip install --no-cache-dir -r requirements-docker.txt

# ── Layer 5: Copy project code ────────────────────────────
# Copies everything from your laptop into /app/
# .dockerignore controls what gets excluded
COPY . .

# ── Layer 6: Create output directories ────────────────────
# These folders need to exist before the pipeline runs
RUN mkdir -p outputs/models \
             outputs/clusters \
             outputs/reports \
             data/raw \
             data/processed \
             reports

# ── Layer 7: Default command ──────────────────────────────
# What runs when someone does: docker run flight-segmentation
# Can be overridden: docker run flight-segmentation python app.py
CMD ["python", "main.py"]