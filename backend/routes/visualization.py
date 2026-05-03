from fastapi import APIRouter, HTTPException
from backend.state import get_df
 
router = APIRouter(prefix="/visualize", tags=["Visualization"])
 
 
@router.get("/")
def visualize():
    try:
        df = get_df()
        numeric = df.select_dtypes(include="number")
 
        return {
            "bar": numeric.iloc[:, 0].value_counts().to_dict(),
            "line": numeric.iloc[:, 0].to_dict(),
            "heatmap": numeric.corr().to_dict()
        }
 
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))