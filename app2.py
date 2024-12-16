from recommender import TravelRecommender
import pandas as pd

posts_df=pd.read_csv("posts_data_preprocessed.csv")
interactions_df=pd.read_csv("interactions_df2.csv")


# Initialize and update models
recommender = TravelRecommender()
recommender.update_models(posts_df, interactions_df)

# Get recommendations (using real user ID)
recommendations = recommender.get_hybrid_recommendations(user_id=68,n_recommendations=20)
print(recommendations)

# Add new interaction (using real IDs)
# success = recommender.add_new_interaction(
#     user_id=566,
#     post_id=811,
#     interaction_type='like'
# )
# if not success:
#     # Trigger full model update to incorporate new user/post
#     recommender.update_models(posts_df, interactions_df)