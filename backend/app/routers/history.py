"""GET /history — paginated list of past predictions (summary view)."""
from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from typing import List

from app.database import get_db
from app.models.db_models import Prediction
from app.models.schemas import HistoryItem

router = APIRouter(tags=["history"])


@router.get("/history", response_model=List[HistoryItem])
def get_history(
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    crop_type: str = Query(default=None),
    db: Session = Depends(get_db),
):
    query = db.query(Prediction).order_by(Prediction.created_at.desc())
    if crop_type:
        query = query.filter(Prediction.crop_type == crop_type)
    rows = query.offset(offset).limit(limit).all()
    return rows
