import requests

# Valid prediction
r = requests.post("http://localhost:8000/predict", json={
    "L": 2500, "R": 15, "F": 180, "M": 420000, "C": 0.95
})
print(r.json())

# Missing field test — expect 422
r = requests.post("http://localhost:8000/predict", json={
    "L": 2500, "R": 15
})
print(r.status_code)  # 422
print(r.json())       # detailed error from Pydantic