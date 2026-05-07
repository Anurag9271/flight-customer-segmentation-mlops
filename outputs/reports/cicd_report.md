# CI/CD Pipeline Report

## CI — GitHub Actions
File: .github/workflows/ci.yml
Trigger: Every push to main branch

Steps:
1. Checkout code
2. Build Docker image from Dockerfile.fastapi
3. Run container on port 8000
4. Test / endpoint → expect 200
5. Test /predict endpoint → expect 200
6. Test validation → expect 422
7. Stop and remove container

## CD — Render Auto Deploy
Platform: Render.com
Trigger: Every push to main branch
Action: Rebuild Docker image and redeploy

## Live URL
https://flight-segmentation-api.onrender.com