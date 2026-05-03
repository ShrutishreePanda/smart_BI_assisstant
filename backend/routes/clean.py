from fastapi import APIRouter, HTTPException
from backend.state import get_df, update_df

router = APIRouter(prefix="/clean", tags=["Cleaning"])


@router.post("/")
def clean_data():
    try:
        df = get_df()

        rows_before = df.shape[0]
        nulls_before = int(df.isna().sum().sum())

        # Remove duplicates
        df = df.drop_duplicates()

        # Fill missing values
        for col in df.columns:
            if df[col].dtype == "object":
                df[col] = df[col].fillna(df[col].mode()[0])
            else:
                df[col] = df[col].fillna(df[col].median())

        rows_after = df.shape[0]
        nulls_after = int(df.isna().sum().sum())

        update_df(df)

        return {
            "rows_before": rows_before,
            "rows_after": rows_after,
            "rows_removed": rows_before - rows_after,
            "nulls_before": nulls_before,
            "nulls_after": nulls_after,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))