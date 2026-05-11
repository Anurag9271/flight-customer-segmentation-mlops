# import requests

# # Valid prediction
# r = requests.post("http://localhost:8000/predict", json={
#     "L": 2500, "R": 15, "F": 180, "M": 420000, "C": 0.95
# })
# print(r.json())

# # Missing field test — expect 422
# r = requests.post("http://localhost:8000/predict", json={
#     "L": 2500, "R": 15
# })
# print(r.status_code)  # 422
# print(r.json())       # detailed error from Pydantic


# import requests

# r = requests.post("http://localhost:8000/predict", json={
#     "L": 2500,
#     "R": 15,
#     "F": 180,
#     "M": 420000,
#     "C": 0.95
# })
# print(r.status_code)
# print(r.json())

# Check which cluster = Champions
# Champions = highest F (flights) and M (km)
import joblib
import pandas as pd
import numpy as np

scaler   = joblib.load('outputs/models/scaler.pkl')
km_model = joblib.load('outputs/models/kmeans_model.pkl')

centers = pd.DataFrame(
    km_model.cluster_centers_,
    columns=['L','R','F','M','C']
)
print("Cluster centers:")
print(centers)
print("\nHighest F (frequency) = Champions cluster:")
print("Champions cluster ID:", centers['F'].idxmax())
print("At-Risk cluster ID (highest R):", centers['R'].idxmax())