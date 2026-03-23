from fastapi import APIRouter
from app.api.schemas.research import CalibrationLatestResponse, CalibrationHistoryResponse

router = APIRouter(prefix="/calibration", tags=["calibration"])

@router.get("/latest", response_model=CalibrationLatestResponse)
def get_latest_calibration():
    # Fetch from ModelCalibration in production
    return CalibrationLatestResponse(
        current_coefficients={"quality": 0.2, "value": 0.2, "momentum": 0.4, "volatility": 0.2},
        calibration_window_start="2023-01-01",
        calibration_window_end="2023-12-31",
        next_update_date="2024-07-01",
        r_squared=0.65
    )

@router.get("/history", response_model=CalibrationHistoryResponse)
def get_calibration_history():
    return CalibrationHistoryResponse(history=[])
