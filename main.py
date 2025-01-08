from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from recommender import TravelRecommender  # Import your recommendation class
import  pandas as pd
from typing import Literal

posts_df=pd.read_csv("posts_data_preprocessed.csv")
interactions_df=pd.read_csv("interactions_df2.csv")

# Initialize FastAPI app
app = FastAPI()

# Initialize the TravelRecommender object
recommender = TravelRecommender()  # Replace with required arguments if any
recommender.update_models(posts_df, interactions_df)

# Request model for the API
class RecommendationRequest(BaseModel):
    user_id: int
    num_recommendations: int = 10  # Default to 5 recommendations

class TrendingRecommendationsRequest(BaseModel):
    num_recommendations: int =10

class AddInteraction(BaseModel):
    user_id:int
    post_id:int
    interaction_type: Literal['like','save','comment','view','share']

# Route to get recommendations
@app.post("/recommend")
def recommend_posts(request: RecommendationRequest):
    """
    Recommend posts using the TravelRecommender class.
    """
    user_id = request.user_id
    num_recommendations = request.num_recommendations


    try:
        # Use the TravelRecommender's recommend method
        recommendations = recommender.get_hybrid_recommendations(user_id, num_recommendations)

        # Check if recommendations are empty
        if not recommendations:
            raise HTTPException(status_code=404, detail="No recommendations found.")

        return {"user_id": user_id, "recommendations": recommendations}

    except Exception as e:
        # Handle errors gracefully
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/Goexplore")
def collaborative(request:RecommendationRequest):
    user_id=request.user_id
    num_recommendations=request.num_recommendations

    try:
        # Use the TravelRecommender's recommend method
        recommendations = recommender.get_collaborative_recommendations(user_id, num_recommendations)

        # Check if recommendations are empty
        if not recommendations:
            raise HTTPException(status_code=404, detail="No recommendations found.")

        return {"user_id": user_id, "recommendations": recommendations}

    except Exception as e:
        # Handle errors gracefully
        raise HTTPException(status_code=500, detail=str(e))

    
@app.post("/trending_posts")
def recommend_trending(request:TrendingRecommendationsRequest):
    n_recommedations=request.num_recommendations
    try:
        recommendations=recommender.get_popular_recommendations(n_recommendations=n_recommedations)
        # Check if recommendations are empty
        if not recommendations:
            raise HTTPException(status_code=404, detail="No recommendations found.")

        return {"recommendations": recommendations}

    except Exception as e:
        # Handle errors gracefully
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/add_interaction")
def add_interaction(request:AddInteraction):
    user_id=request.user_id
    post_id=request.post_id
    interaction_type=request.interaction_type
    
    try:
        success=recommender.add_new_interaction(user_id,post_id,interaction_type)
        # Check if recommendations are empty
        if not success:
            raise HTTPException(status_code=404, detail="No recommendations found.")

        return {"Status":"success"}

    except Exception as e:
        # Handle errors gracefully
        raise HTTPException(status_code=500, detail=str(e))



# Test route
@app.get("/")
def root():
    return {"message": "Recommendation API is running!"}



