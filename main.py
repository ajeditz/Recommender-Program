# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# # from recommender import TravelRecommender  # Import your recommendation class
# import  pandas as pd
# from pilot_recommender import TravelRecommender
# from typing import Literal

# posts_df=pd.read_csv("pilot_data_preprocessed_new.csv")
# interactions_df=pd.read_csv("pilot_interaction.csv")
# connections_df = pd.read_csv("connections.csv")  # New: Load connections data


# # Initialize FastAPI app
# app = FastAPI()

# # Initialize the TravelRecommender object
# recommender = TravelRecommender()  # Replace with required arguments if any
# recommender.update_models(posts_df, interactions_df)
# recommender.set_user_connections(connections_df)  # Set up user connections


# # Request model for the API
# class RecommendationRequest(BaseModel):
#     user_id: int
#     num_recommendations: int = 10  # Default to 5 recommendations
#     # connection_ratio:float
#     # max_post_age_days:int=30

# class TrendingRecommendationsRequest(BaseModel):
#     num_recommendations: int =10

# class AddInteraction(BaseModel):
#     user_id:int
#     post_id:int
#     interaction_type: Literal['like','save','comment','view','share']

# # Route to get recommendations
# @app.post("/recommend")
# def recommend_posts(request: RecommendationRequest):
#     """
#     Recommend posts using the TravelRecommender class.
#     """
#     user_id = request.user_id
#     num_recommendations = request.num_recommendations
#     # connection_ratio=request.connection_ratio
#     # max_post_age_days=request.max_post_age_days


#     try:
#         # Use the TravelRecommender's recommend method
#         recommendations = recommender.get_hybrid_recommendations(user_id, num_recommendations)

#         # Check if recommendations are empty
#         if not recommendations:
#             raise HTTPException(status_code=404, detail="No recommendations found.")

#         return {"user_id": user_id, "recommendations": recommendations}

#     except Exception as e:
#         # Handle errors gracefully
#         raise HTTPException(status_code=500, detail=str(e))
    
# @app.post("/Goexplore")
# def collaborative(request:RecommendationRequest):
#     user_id=request.user_id
#     num_recommendations=request.num_recommendations

#     try:
#         # Use the TravelRecommender's recommend method
#         recommendations = recommender.get_collaborative_recommendations(user_id, num_recommendations)

#         # Check if recommendations are empty
#         if not recommendations:
#             raise HTTPException(status_code=404, detail="No recommendations found.")

#         return {"user_id": user_id, "recommendations": recommendations}

#     except Exception as e:
#         # Handle errors gracefully
#         raise HTTPException(status_code=500, detail=str(e))

    
# @app.post("/trending_posts")
# def recommend_trending(request:TrendingRecommendationsRequest):
#     n_recommedations=request.num_recommendations
#     try:
#         recommendations=recommender.get_popular_recommendations(n_recommendations=n_recommedations)
#         # Check if recommendations are empty
#         if not recommendations:
#             raise HTTPException(status_code=404, detail="No recommendations found.")

#         return {"recommendations": recommendations}

#     except Exception as e:
#         # Handle errors gracefully
#         raise HTTPException(status_code=500, detail=str(e))
    
# @app.post("/add_interaction")
# def add_interaction(request:AddInteraction):
#     user_id=request.user_id
#     post_id=request.post_id
#     interaction_type=request.interaction_type
    
#     try:
#         success=recommender.add_new_interaction(user_id,post_id,interaction_type)
#         # Check if recommendations are empty
#         if not success:
#             raise HTTPException(status_code=404, detail="No recommendations found.")

#         return {"Status":"success"}

#     except Exception as e:
#         # Handle errors gracefully
#         raise HTTPException(status_code=500, detail=str(e))




# # Test route
# @app.get("/")
# def root():
#     return {"message": "Recommendation API is running!"}



# **********************************************************************************************************************************


from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Literal, Optional
import pandas as pd
from pilot_recommender_1 import TravelRecommender

class RecommendationRequest(BaseModel):
    user_id: int = Field(..., description="User ID to get recommendations for")
    num_recommendations: int = Field(default=10, ge=1, le=50, description="Number of recommendations to return")
    connection_ratio: Optional[float] = Field(
        default=None, 
        ge=0.0, 
        le=1.0, 
        description="Ratio of connection posts to include (0.0 to 1.0). If None, prioritizes all connection posts first"
    )
    max_post_age_days: Optional[int] = Field(
        default=30, 
        ge=1, 
        description="Maximum age of connection posts in days"
    )
    detailed_response:bool =False

class TrendingRecommendationsRequest(BaseModel):
    num_recommendations: int = Field(
        default=10, 
        ge=1, 
        le=50, 
        description="Number of trending recommendations to return"
    )
    detailed_response:bool =False

class AddInteraction(BaseModel):
    user_id: int = Field(..., description="User ID performing the interaction")
    post_id: int = Field(..., description="Post ID being interacted with")
    interaction_type: Literal['like', 'save', 'comment', 'view', 'share'] = Field(
        ..., 
        description="Type of interaction"
    )

# Initialize FastAPI app
app = FastAPI(
    title="Travel Post Recommendation API",
    description="API for getting personalized travel post recommendations",
    version="1.0.0"
)

# Initialize data and recommender
try:
    posts_df = pd.read_csv("pilot_data_preprocessed_new.csv")
    interactions_df = pd.read_csv("pilot_interaction.csv")
    connections_df = pd.read_csv("connections.csv")
    
    recommender = TravelRecommender()
    recommender.update_models(posts_df, interactions_df)
    recommender.set_user_connections(connections_df)
    
except Exception as e:
    print(f"Error initializing recommender: {str(e)}")
    raise

@app.post("/recommend", 
          response_model_exclude_none=True,
          summary="Get personalized recommendations",
          response_description="List of recommended posts")
async def recommend_posts(request: RecommendationRequest):
    """
    Get personalized post recommendations for a specific user.
    
    - Supports controlling the ratio of connection vs non-connection posts
    - Can filter connection posts by age
    - Falls back to collaborative and content-based recommendations if needed
    """
    try:
        recommendations = recommender.get_hybrid_recommendations(
            user_id=request.user_id,
            n_recommendations=request.num_recommendations,
            connection_ratio=request.connection_ratio,
            max_post_age_days=request.max_post_age_days,
            detailed_response=request.detailed_response
        )

        if not recommendations:
            raise HTTPException(
                status_code=404,
                detail=f"No recommendations found for user {request.user_id}"
            )

        return {
            "user_id": request.user_id,
            "num_recommendations": len(recommendations),
            "recommendations": recommendations
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# @app.post("/explore",  # Changed from "Goexplore" for consistency
#           response_model_exclude_none=True,
#           summary="Get collaborative recommendations")
# async def get_collaborative_recommendations(request: RecommendationRequest):
#     """Get recommendations based on collaborative filtering only"""
#     try:
#         recommendations = recommender.get_collaborative_recommendations(
#             user_id=request.user_id,
#             n_recommendations=request.num_recommendations,
#             detailed_respons=request.detailed_response
#         )

#         if not recommendations:
#             raise HTTPException(
#                 status_code=404,
#                 detail=f"No collaborative recommendations found for user {request.user_id}"
#             )

#         return {
#             "user_id": request.user_id,
#             "num_recommendations": len(recommendations),
#             "recommendations": recommendations
#         }
#     except ValueError as e:
#         raise HTTPException(status_code=400, detail=str(e))
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.post("/explore",
          response_model_exclude_none=True,
          summary="Get collaborative recommendations")
async def get_collaborative_recommendations(request: RecommendationRequest):
    """Get recommendations based on collaborative filtering only"""
    try:
        recommendations = recommender.get_collaborative_recommendations(
            user_id=request.user_id,
            n_recommendations=request.num_recommendations,
            detailed_response=request.detailed_response  # Fixed typo from 'detailed_respons'
        )

        if not recommendations:
            raise HTTPException(
                status_code=404,
                detail=f"No collaborative recommendations found for user {request.user_id}"
            )

        return {
            "user_id": request.user_id,
            "num_recommendations": len(recommendations),
            "recommendations": recommendations
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/trending",  # Changed from "trending_posts" for consistency
          response_model_exclude_none=True,
          summary="Get trending posts")
async def get_trending_posts(request: TrendingRecommendationsRequest):
    """Get current trending posts based on popularity and recency"""
    try:
        recommendations = recommender.get_popular_recommendations(
            n_recommendations=request.num_recommendations,
            detailed_response=request.detailed_response
        )

        if not recommendations:
            raise HTTPException(status_code=404, detail="No trending posts found")

        return {
            "num_recommendations": len(recommendations),
            "recommendations": recommendations
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/interactions",  # Changed from "add_interaction" for consistency
          response_model_exclude_none=True,
          summary="Record a user interaction with a post")
async def add_interaction(request: AddInteraction):
    """Record a new user interaction (like, save, comment, etc.) with a post"""
    try:
        success = recommender.add_new_interaction(
            request.user_id,
            request.post_id,
            request.interaction_type
        )

        if not success:
            raise HTTPException(
                status_code=400,
                detail="Failed to record interaction"
            )

        return {"status": "success", "message": "Interaction recorded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/health")  # Changed from root() for clarity
async def health_check():
    """Check if the API is running"""
    return {
        "status": "healthy",
        "message": "Recommendation API is running"
    }


if __name__=="__main__":
    import uvicorn
    uvicorn.run("main:app",host="127.0.0.1",port=8000,reload=True)