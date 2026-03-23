from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os

from app.api.routers import stocks, portfolios, research, calibration, governance
from app.assistant.router import router as assistant_router

app = FastAPI(
    title="AI Stock Engine API",
    description="Clean, modular, production-ready API for the full AI Stock Engine.",
    version="1.0.0"
)

# Mount Static Files
static_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "static")
if os.path.exists(static_dir):
    app.mount("/assistant", StaticFiles(directory=static_dir, html=True), name="assistant")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(stocks.router, prefix="/api/v1")
app.include_router(portfolios.router, prefix="/api/v1")
app.include_router(research.router, prefix="/api/v1")
app.include_router(calibration.router, prefix="/api/v1")
app.include_router(governance.router, prefix="/api/v1")
app.include_router(assistant_router, prefix="/api/v1")

@app.get("/health")
@app.get("/api/v1/health")
def health_check():
    from services.portfolio import get_latest_scoring_date
    latest = get_latest_scoring_date()
    
    # Stubbed calibration date (connect to actual DB/Calibration later)
    last_calibration = "2024-01-01"
    
    return {
        "status": "ok",
        "engine": "ready",
        "last_calibration": last_calibration,
        "last_rebalance": latest.isoformat() if latest else "N/A"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
