import uvicorn
import pickle
from fastapi import FastAPI
from Banknote import BankNote

app = FastAPI()

with open("classifier.pkl", "rb") as pickle_in:
    classifier = pickle.load(pickle_in)

@app.post("/predict")
def predict_banknote(data: BankNote):
    prediction = classifier.predict([[
        data.variance,
        data.skewness,
        data.curtosis,
        data.entropy
    ]])

    result = int(prediction[0])

    if result == 1:
        return {"prediction": "Authentic Banknote"}
    else:
        return {"prediction": "Fake Banknote"}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=5000)