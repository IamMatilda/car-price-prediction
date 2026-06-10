from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI()

pipeline = joblib.load("models/pipeline.pkl")

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth")


class CarFeatures(BaseModel):
    Brand: str
    model: str
    Year: int
    kmDriven: float
    Transmission: str
    Owner: str
    FuelType: str


async def is_token_correct(token: str) -> bool:
    return token == "00000"


async def check_token(token: str = Depends(oauth2_scheme)) -> None:
    if not await is_token_correct(token):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )


@app.get("/healthcheck")
def healthcheck():
    return {"status": "ok"}


@app.post("/predictions")
async def predictions(
    instance: CarFeatures,
    token: str = Depends(check_token)
):
    df = pd.DataFrame([instance.dict()])

    prediction = pipeline.predict(df)

    return {
        "predicted_price": float(prediction[0])
    }