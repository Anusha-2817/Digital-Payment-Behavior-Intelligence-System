from fastapi import FastAPI

from app.api.routes.predict import router

app = FastAPI(
    title="Behavioral Intelligence API",
    version="1.0"
)

app.include_router(router)


@app.get("/")
def home():

    return {
        "message": (
            "Behavioral Intelligence API Running"
        )
    }


@app.get("/health")
def health_check():

    return {
        "status": "healthy"
    }