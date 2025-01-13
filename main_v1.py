from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from typing import Literal, List, Optional
import pandas as pd
from pilot_recommender import TravelRecommender
from datetime import datetime

app = FastAPI(title="Travel Recommender API",
             description="API for travel content recommendations and interactions")

# Load data
posts_df = pd.read_csv("pilot_data_preprocessed_new.csv")
interactions_df = pd.read_csv("pilot_interaction.csv")
connections_df = pd.read_csv("connections.csv")  # New: Load connections data

# Initialize recommender
recommender = TravelRecommender()
recommender.update_models(posts_df, interactions_df)
recommender.set_user_connections(connections_df)  # Set up user connections

# Request/Response Models
class RecommendationRequest(BaseModel):
    user_id: int = Field(..., description="User ID to get recommendations for")
    num_recommendations: int = Field(default=10, ge=1, le=50, description="Number of recommendations to return")

class TrendingRecommendationsRequest(BaseModel):
    num_recommendations: int = Field(default=10, ge=1, le=50, description="Number of trending posts to return")
    max_age_days: Optional[int] = Field(default=30, ge=1, description="Maximum age of posts in days")

class AddInteraction(BaseModel):
    user_id: int = Field(..., description="User ID performing the interaction")
    post_id: int = Field(..., description="Post ID being interacted with")
    interaction_type: Literal['like', 'save', 'comment', 'view', 'share'] = Field(
        ..., description="Type of interaction")

class RecommendationResponse(BaseModel):
    post_id: int
    score: float
    types: List[str]
    from_connection: Optional[bool]
    recency_score: Optional[float]
    popularity_score: Optional[float]

class RecommendationsListResponse(BaseModel):
    user_id: Optional[int]
    recommendations: List[RecommendationResponse]
    total_count: int
    generated_at: datetime

# Enhanced API endpoints
@app.post("/recommend", response_model=RecommendationsListResponse)
async def recommend_posts(request: RecommendationRequest):
    """
    Get personalized post recommendations for a user's timeline.
    Prioritizes posts from user's connections and content similar to their interests.
    """
    try:
        recommendations = recommender.get_hybrid_recommendations(
            user_id=request.user_id,
            n_recommendations=request.num_recommendations
        )

        if not recommendations:
            raise HTTPException(
                status_code=404,
                detail="No recommendations found for this user."
            )

        return RecommendationsListResponse(
            user_id=request.user_id,
            recommendations=recommendations,
            total_count=len(recommendations),
            generated_at=datetime.now()
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/explore", response_model=RecommendationsListResponse)
async def get_explore_recommendations(request: RecommendationRequest):
    """
    Get recommendations for the GoExplore feature.
    Shows content from users similar to the requester but not necessarily from connections.
    """
    try:
        recommendations = recommender.get_collaborative_recommendations(
            user_id=request.user_id,
            n_recommendations=request.num_recommendations
        )

        if not recommendations:
            raise HTTPException(
                status_code=404,
                detail="No explore recommendations found for this user."
            )

        return RecommendationsListResponse(
            user_id=request.user_id,
            recommendations=recommendations,
            total_count=len(recommendations),
            generated_at=datetime.now()
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/trending", response_model=RecommendationsListResponse)
async def get_trending_posts(request: TrendingRecommendationsRequest):
    """
    Get trending posts with recency factor.
    Returns popular posts weighted by both engagement metrics and recency.
    """
    try:
        recommendations = recommender.get_popular_recommendations(
            n_recommendations=request.num_recommendations
        )

        if not recommendations:
            raise HTTPException(
                status_code=404,
                detail="No trending posts found."
            )

        return RecommendationsListResponse(
            recommendations=recommendations,
            total_count=len(recommendations),
            generated_at=datetime.now()
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/interaction")
async def add_interaction(interaction: AddInteraction):
    """
    Record a new user interaction with a post.
    Updates the recommendation models and invalidates relevant caches.
    """
    try:
        success = recommender.add_new_interaction(
            user_id=interaction.user_id,
            post_id=interaction.post_id,
            interaction_type=interaction.interaction_type
        )

        return {
            "status": "success",
            "message": "Interaction recorded successfully",
            "timestamp": datetime.now()
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to record interaction: {str(e)}"
        )

# Health check endpoint
@app.get("/health")
async def health_check():
    """
    Check API health and basic system status.
    """
    return {
        "status": "healthy",
        "timestamp": datetime.now(),
        "recommender_status": {
            "models_loaded": bool(recommender.post_vectors is not None),
            "last_update": recommender.redis_client.get('last_update_timestamp')
        }
    }