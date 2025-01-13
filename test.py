from pilot_recommender import TravelRecommender
import pandas as pd
from datetime import datetime

def load_and_verify_data():
    """Load and verify the data files"""
    try:
        posts_df = pd.read_csv("pilot_data_preprocessed_new.csv")
        interactions_df = pd.read_csv("pilot_interaction.csv")
        connections_df = pd.read_csv("connections.csv")
        
        print("Data loaded successfully:")
        print(f"Posts shape: {posts_df.shape}")
        print(f"Interactions shape: {interactions_df.shape}")
        print(f"Connections shape: {connections_df.shape}\n")
        
        return posts_df, interactions_df, connections_df
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None, None, None

def test_recommendations(user_id=1, n_recommendations=5):
    """Test the recommender system with loaded data"""
    try:
        # Load data
        posts_df, interactions_df, connections_df = load_and_verify_data()
        if posts_df is None:
            return
            
        # Initialize recommender
        recommender = TravelRecommender()
        recommender.update_models(posts_df, interactions_df)
        
        # Get recommendations
        test_recommendations = recommender.get_collaborative_recommendations(
            user_id=user_id, 
            n_recommendations=n_recommendations
        )
        
        # Print recommendations with proper timezone handling
        print(f"\nTesting recommendations for user_id: {user_id}")
        print("=" * 50)
        
        for rec in test_recommendations:
            # Get post details
            post = recommender.post_features[
                recommender.post_features['post_id'] == rec['post_id']
            ].iloc[0]
            
            # Handle timezone-naive datetime conversion
            created_at = pd.to_datetime(post['created_at'])
            if created_at.tzinfo is not None:
                created_at = created_at.tz_localize(None)
            
            now = pd.Timestamp.now().tz_localize(None)
            age_days = (now - created_at).days
            
            # Calculate expected recency score
            expected_recency = max(0, 1 - (age_days / 30)) if age_days <= 30 else 0
            
            # Print detailed information
            print(f"\nPost {rec['post_id']}:")
            print(f"Created: {created_at}")
            print(f"Age: {age_days} days")
            print(f"Recency score: {rec['recency_score']:.4f}")
            print(f"Expected recency: {expected_recency:.4f}")
            print(f"Total score: {rec['score']:.4f}")
            
            # Alert if recency score doesn't match expected
            if abs(rec['recency_score'] - expected_recency) > 0.01:
                print("WARNING: Recency score differs from expected value!")
        
    except Exception as e:
        print(f"Error during testing: {str(e)}")

if __name__ == "__main__":
    # Run the test
    test_recommendations(user_id=18, n_recommendations=5)