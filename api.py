from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import logging
import uvicorn  # Required to run the app

# Assumes recommendation_engine.py is in the same directory or Python path
try:
    from recommendation_engine import AssessmentRecommender
except ImportError:
    logging.error(
        "Failed to import AssessmentRecommender. Ensure recommendation_engine.py is accessible."
    )
    # Exit or raise a more specific error if the engine is critical
    raise SystemExit("AssessmentRecommender could not be imported.")


# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# --- FastAPI App Initialization ---
app = FastAPI(
    title="SHL Assessment Recommender API",
    description="Recommends SHL assessments based on natural language queries or job descriptions.",
    version="1.0.0",
)


# --- Data Models ---
class Query(BaseModel):
    text: str = Field(
        ..., description="Natural language query or job description text."
    )
    top_k: Optional[int] = Field(
        10, description="Maximum number of recommendations to return.", gt=0, le=50
    )  # Added top_k param


# Define the structure of a single recommendation item in the response
class RecommendationItem(BaseModel):
    name: Optional[str] = None
    url: Optional[str] = None
    remote_testing: Optional[str] = None
    adaptive_support: Optional[str] = None
    duration_minutes: Optional[float] = (
        None  # Allow float due to potential NaNs becoming None
    )
    test_type: Optional[str] = None
    # Add other relevant fields from the recommender if needed
    # description: Optional[str] = None
    # job_levels: Optional[str] = None


class RecommendationResponse(BaseModel):
    results: List[RecommendationItem]


# --- Global Recommender Instance ---
# Initialize the recommender once when the application starts
# Handle potential initialization errors gracefully
try:
    # Make sure the data path used here matches the output of preprocessing
    recommender = AssessmentRecommender(data_path="./data/processed_assessments.csv")
    logging.info("AssessmentRecommender initialized successfully for API.")
except Exception as e:
    logging.exception("FATAL: Failed to initialize AssessmentRecommender for API.")
    # The application might not be usable without the recommender.
    # Consider exiting or letting FastAPI handle the lack of the object later.
    recommender = None  # Set to None to indicate failure


# --- API Endpoints ---
@app.get("/")
async def read_root():
    """Basic endpoint to check if the API is running."""
    return {"message": "SHL Assessment Recommender API is running."}


@app.post("/recommend", response_model=RecommendationResponse)
async def recommend_assessments(query: Query):
    """
    Recommends SHL assessments based on the provided query text.
    """
    if recommender is None:
        logging.error("Recommender not available.")
        raise HTTPException(
            status_code=503, detail="Recommendation engine is not initialized."
        )

    logging.info(
        f"Received recommendation request: top_k={query.top_k}, query='{query.text[:100]}...'"
    )

    try:
        # Call the recommendation engine
        results_list = recommender.recommend(query.text, top_k=query.top_k)
        logging.info(f"Recommendation engine returned {len(results_list)} results.")

        # Ensure the results match the Pydantic model structure
        # The recommender should return a list of dicts matching RecommendationItem fields
        # No specific transformation needed here if recommender returns the correct dicts

        return {"results": results_list}

    except Exception as e:
        # Log the full error for debugging
        logging.exception(
            f"Error processing recommendation request for query '{query.text[:100]}...': {e}"
        )
        # Return a generic server error to the client
        raise HTTPException(
            status_code=500, detail="Internal server error during recommendation."
        )


# --- Main Execution (for running with uvicorn) ---
if __name__ == "__main__":
    # Run using: uvicorn api:app --reload
    logging.info("Starting FastAPI server with Uvicorn...")
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
    # The reload=True flag is useful for development
