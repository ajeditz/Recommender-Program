from recommender import TravelRecommender
import pandas as pd

posts_df=pd.read_csv("new_df.csv")
interactions_df=pd.read_csv("interactions_df2.csv")

# Initialize the recommender
recommender = TravelRecommender()

# Update models with your data
# recommender.update_models(posts_df, interactions_df)

# Get recommendations for a user
recommendations = recommender.get_hybrid_recommendations(user_id=1012, n_recommendations=10)
print(recommendations)
# Process new interaction
# recommender.add_new_interaction(user_id=123, post_id=456, interaction_type='like')