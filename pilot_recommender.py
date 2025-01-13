import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import redis
import json
from datetime import datetime
from typing import List, Dict, Tuple,Optional,Union
import logging
from datetime import datetime



# class TravelRecommender:
#     def __init__(self, redis_host='localhost', redis_port=6379):
#         # Initialize Redis for caching
#         self.redis_client = redis.Redis(host=redis_host, port=redis_port)
#         self.cache_ttl = 3600  # Cache TTL in seconds
        
#         # Initialize storage for our models
#         self.post_vectors = None
#         self.user_item_matrix = None
#         self.post_features = None
#         self.similarity_matrix = None
#         self.user_connections = None  # New: Store user connections
        
#         # Configure logging
#         logging.basicConfig(level=logging.INFO)
#         self.logger = logging.getLogger(__name__)



class TravelRecommender:
    def __init__(self, redis_host='localhost', redis_port=6379):
        # Initialize Redis for caching
        self.redis_client = redis.Redis(host=redis_host, port=redis_port)
        self.cache_ttl = 3600  # Cache TTL in seconds
        
        # Initialize storage for our models
        self.post_vectors = None
        self.user_item_matrix = None
        self.post_features = None
        self.similarity_matrix = None
        self.user_connections = None
        self.user_index_mapping = {}  # New: Add user index mapping
        
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    # def set_user_connections(self, connections_df: pd.DataFrame):
    #     """
    #     Set up user connections mapping
    #     """
    #     self.user_connections = connections_df.set_index('user_id')['connections'].to_dict()


    def set_user_connections(self, connections_df: pd.DataFrame):
        """
        Set up user connections mapping with proper parsing of connection strings
        
        Parameters:
        - connections_df: DataFrame with 'user_id' and 'connections' columns
        """
        try:
            def parse_connections(conn_str):
                # Handle string representation of list
                if isinstance(conn_str, str):
                    # Remove brackets, spaces, and split by commas
                    cleaned = conn_str.strip('[]').replace(' ', '')
                    # Split by comma and convert to integers, filtering out empty strings
                    return [int(x) for x in cleaned.split(',') if x]
                # Handle if it's already a list
                elif isinstance(conn_str, list):
                    return conn_str
                # Handle NaN or other invalid values
                return []

            # Create the connections dictionary
            self.user_connections = {}
            for _, row in connections_df.iterrows():
                user_id = row['user_id']
                connections = parse_connections(row['connections'])
                self.user_connections[user_id] = connections
                
            self.logger.info(f"Successfully processed connections for {len(self.user_connections)} users")
            
        except Exception as e:
            self.logger.error(f"Error setting user connections: {str(e)}")
            self.user_connections = {}


    def get_user_connections(self, user_id: int) -> List[int]:
        """
        Get list of connections for a user
        
        Parameters:
        - user_id: ID of the user
        
        Returns:
        - List of user IDs representing connections
        """
        return self.user_connections.get(user_id, [])



    def make_naive(self,timezone_aware_dt):

        """
        Converts a timezone-aware datetime object to a timezone-naive datetime object.

        Parameters:
            timezone_aware_dt (datetime): A timezone-aware datetime object.

        Returns:
            datetime: A timezone-naive datetime object.
        """
        if timezone_aware_dt.tzinfo is None:
            raise ValueError("The provided datetime is not timezone-aware.")
        
        # Remove timezone information
        naive_dt = timezone_aware_dt.astimezone(None).replace(tzinfo=None)
        return naive_dt


    # def calculate_recency_score(self, post_date: datetime, max_age_days: int = 30) -> float:
    #     """
    #     Calculate recency score for a post
    #     Returns a score between 0 and 1, where 1 is most recent
    #     """
    #     now = datetime.now()
    #     print(type(now))
    #     age_days = (now - post_date).days
        
    #     if age_days < 0:
    #         return 0.0
        
    #     # Linear decay over max_age_days
    #     score = max(0, 1 - (age_days / max_age_days))
    #     return score


    # def calculate_recency_score(self, post_date: str, max_age_days: int = 30) -> float:
    #     """
    #     Calculate recency score for a post with proper timezone handling
    #     """
    #     try:
    #         # Convert to timezone-naive datetime if necessary
    #         if isinstance(post_date, str):
    #             post_date = pd.to_datetime(post_date).replace(tzinfo=None)
    #         elif hasattr(post_date, 'tzinfo') and post_date.tzinfo is not None:
    #             post_date = post_date.replace(tzinfo=None)
                
    #         now = datetime.now()
    #         age_days = (now - post_date).days
    #         print(f"this is age dys{age_days}")
            
    #         if age_days < 0:
    #             return 0.0
            
    #         return max(0, 1 - (age_days / max_age_days))
    #     except Exception as e:
    #         self.logger.error(f"Error calculating recency score: {str(e)}")
    #         return 0.0


    def calculate_recency_score(self, post_date: str, max_age_days: int = 60) -> float:
        """
        Calculate recency score for a post with proper timezone handling
        """
        try:
            # Convert string to datetime if needed
            if isinstance(post_date, str):
                post_date = pd.to_datetime(post_date)
                
            # Ensure we're working with timezone-naive datetimes
            if post_date.tzinfo is not None:
                post_date = post_date.tz_localize(None)
                
            now = pd.Timestamp.now()
            
            # Calculate age in days
            age_days = (now - post_date).days
            
            # Debug logging
            self.logger.debug(f"Post date: {post_date}, Now: {now}, Age days: {age_days}")
            
            # Handle edge cases
            if age_days < 0:
                self.logger.warning(f"Negative age detected for post date {post_date}")
                return 0.0
                
            if age_days > max_age_days:
                return 0.0
                
            # Calculate recency score
            recency_score = 1 - (age_days / max_age_days)
            
            self.logger.debug(f"Calculated recency score: {recency_score}")
            
            return max(0.0, min(1.0, recency_score))  # Ensure score is between 0 and 1
            
        except Exception as e:
            self.logger.error(f"Error calculating recency score: {str(e)}")
            return 0.0





    def process_post_content(self, posts_df: pd.DataFrame) -> np.ndarray:

        """
        Process post content using TF-IDF vectorization
        """
        try:
            # Print columns for debugging
            self.logger.info(f"Available columns: {posts_df.columns}")
            
            # Combine relevant text fields with proper null handling
            posts_df['combined_text'] = (
                posts_df['title'].fillna('') + ' ' +
                posts_df['description'].fillna('') + ' ' +
                posts_df['tags'].fillna('')
            ).str.strip()
            
            # Create TF-IDF vectors
            vectorizer = TfidfVectorizer(
                max_features=100,
                stop_words='english',
                ngram_range=(1, 2)
            )
            
            # Convert any non-string values to strings
            posts_df['combined_text'] = posts_df['combined_text'].astype(str)
            
            vector = vectorizer.fit_transform(posts_df['combined_text'])
            return vector
        
        except Exception as e:
            self.logger.error(f"Error in process_post_content: {str(e)}")
            raise
    

    # def process_post_content(self, posts_df: pd.DataFrame) -> np.ndarray:
    #     """
    #     Process post content using TF-IDF vectorization
    #     """
    #     # Combine relevant text fields
    #     print(posts_df.columns)
    #     posts_df['combined_text'] = posts_df['']+" "+posts_df['description']+" "+posts_df['tags']+" "+ posts_df['location']
    #     # posts_df['combined_text']= posts_df['combined_text'].apply(lambda x: " ".join(x))
        
        
    #     # Create TF-IDF vectors
    #     vectorizer = TfidfVectorizer(
    #         max_features=100,
    #         stop_words='english',
    #         ngram_range=(1, 2)
    #     )
    #     # print(vectorizer)
    #     vector= vectorizer.fit_transform(posts_df['combined_text'])
    #     # print vectorizer.get_feature_names()
    #     return vector

    # def build_user_item_matrix(self, interactions_df: pd.DataFrame) -> np.ndarray:
    #     """
    #     Build user-item interaction matrix with weights
    #     """
    #     # Define interaction weights
    #     weights = {
    #         'view': 1,
    #         'like': 3,
    #         'comment': 4,
    #         'share': 5,
    #         'save':3
    #     }
        
    #     # Create pivot table with weighted interactions
    #     matrix = pd.pivot_table(
    #         interactions_df,
    #         values='interaction_type',
    #         index='user_id',
    #         columns='post_id',
    #         aggfunc=lambda x: sum(weights.get(i, 1) for i in x),
    #         fill_value=0
    #     )
        
    #     return matrix.values




    def build_user_item_matrix(self, interactions_df: pd.DataFrame) -> np.ndarray:
        """
        Build user-item interaction matrix with proper index mapping
        """
        try:
            # Create user index mapping
            unique_users = sorted(interactions_df['user_id'].unique())
            self.user_index_mapping = {user_id: idx for idx, user_id in enumerate(unique_users)}
            
            # Define interaction weights
            weights = {
                'view': 1,
                'like': 3,
                'comment': 4,
                'share': 5,
                'save': 3
            }
            
            # Map user_ids to indices
            interactions_df['user_idx'] = interactions_df['user_id'].map(self.user_index_mapping)
            
            # Create pivot table with weighted interactions
            matrix = pd.pivot_table(
                interactions_df,
                values='interaction_type',  # Changed from interaction_type to activity_type
                index='user_idx',
                columns='post_id',  # Changed from post_id to content_id
                aggfunc=lambda x: sum(weights.get(i, 1) for i in x),
                fill_value=0
            )
            
            return matrix.values
            
        except Exception as e:
            self.logger.error(f"Error building user-item matrix: {str(e)}")
            return np.array([])
        
    def get_user_matrix_index(self, user_id: int) -> int:
        """
        Get the correct matrix index for a user_id
        """
        try:
            return self.user_index_mapping[user_id]
        except KeyError:
            raise ValueError(f"User ID {user_id} not found in interaction history")


    
    def update_models(self, posts_df: pd.DataFrame, interactions_df: pd.DataFrame):
        """
        Update all models with new data
        """
        try:
            self.logger.info("Starting model update...")
            
            # Update content-based features
            self.post_vectors = self.process_post_content(posts_df)
            self.post_features = posts_df
            
            # Update collaborative filtering matrix
            self.user_item_matrix = self.build_user_item_matrix(interactions_df)
            
            # Calculate post similarity matrix
            self.similarity_matrix = cosine_similarity(self.post_vectors)
            
            # Cache the timestamp of last update
            self.redis_client.set('last_update_timestamp', str(datetime.now()))
            
            self.logger.info("Model update completed successfully")
            
        except Exception as e:
            self.logger.error(f"Error updating models: {str(e)}")
            raise
    



    
    # def get_content_based_recommendations(self, user_id: int, n_recommendations: int = 5) -> List[Dict]:
    #     """
    #     Get content-based recommendations based on user's interaction history
    #     """
    #     # Check cache first
    #     cache_key = f"content_recs:{user_id}"
    #     cached_recs = self.redis_client.get(cache_key)
    #     if cached_recs:
    #         return json.loads(cached_recs)

    #     try:
    #         # Get user's interaction history
    #         user_interactions = self.user_item_matrix[user_id]
            
    #         # Find posts user has interacted with
    #         interacted_posts = np.where(user_interactions > 0)[0]
            
    #         if len(interacted_posts) == 0:
    #             return self.get_popular_recommendations(n_recommendations)
            
    #         # Calculate average similarity with interacted posts
    #         sim_scores = np.mean([self.similarity_matrix[i] for i in interacted_posts], axis=0)
            
    #         # Get top similar posts
    #         similar_posts = np.argsort(sim_scores)[::-1]
            
    #         # Filter out already interacted posts
    #         recommendations = [
    #             {
    #                 'post_id': int(self.post_features.iloc[i]['post_id']),
    #                 'score': float(sim_scores[i]),
    #                 'type': 'content'
    #             }
    #             for i in similar_posts
    #             if i not in interacted_posts
    #         ][:n_recommendations]
            
    #         # Cache the results
    #         self.redis_client.setex(cache_key, self.cache_ttl, json.dumps(recommendations))
            
    #         return recommendations
            
    #     except Exception as e:
    #         self.logger.error(f"Error in content-based recommendations: {str(e)}")
    #         return self.get_popular_recommendations(n_recommendations)



    def get_content_based_recommendations(self, user_id: int, n_recommendations: int = 5) -> List[Dict]:
        """
        Get content-based recommendations with proper index handling
        """
        try:
            # Get correct user index
            user_idx = self.get_user_matrix_index(user_id)
            
            # Get user's interaction history
            user_interactions = self.user_item_matrix[user_idx]
            
            # Find posts user has interacted with
            interacted_posts = np.where(user_interactions > 0)[0]
            
            if len(interacted_posts) == 0:
                return self.get_popular_recommendations(n_recommendations)
            
            # Calculate average similarity with interacted posts
            sim_scores = np.mean([self.similarity_matrix[i] for i in interacted_posts], axis=0)
            
            # Get top similar posts
            similar_posts = np.argsort(sim_scores)[::-1]
            
            # Make sure we don't go out of bounds
            valid_indices = [i for i in similar_posts if i < len(self.post_features)]
            
            # Filter out already interacted posts
            recommendations = [
                {
                    'post_id': int(self.post_features.iloc[i]['post_id']),
                    'score': float(sim_scores[i]),
                    'type': 'content'
                }
                for i in valid_indices
                if i not in interacted_posts
            ][:n_recommendations]
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error in content-based recommendations: {str(e)}")
            return self.get_popular_recommendations(n_recommendations)



    # def get_collaborative_recommendations(self, user_id: int, n_recommendations: int = 5) -> List[Dict]:
    #     """
    #     Get collaborative filtering recommendations using user similarity
    #     """
    #     try:
    #         # Calculate user similarity
    #         user_similarity = cosine_similarity([self.user_item_matrix[user_id]], self.user_item_matrix)[0]
            
    #         # Get most similar users
    #         similar_users = np.argsort(user_similarity)[::-1][1:6]  # Get top 5 similar users
            
    #         # Get their highly rated posts
    #         similar_user_posts = defaultdict(float)
            
    #         for sim_user_idx in similar_users:
    #             sim_score = user_similarity[sim_user_idx]
    #             user_ratings = self.user_item_matrix[sim_user_idx]
                
    #             for post_idx, rating in enumerate(user_ratings):
    #                 if rating > 0:
    #                     similar_user_posts[post_idx] += rating * sim_score
            
    #         # Sort and filter recommendations
    #         recommendations = [
    #             {
    #                 'post_id': int(self.post_features.iloc[post_idx]['post_id']),
    #                 'score': float(score),
    #                 'type': 'collaborative'
    #             }
    #             for post_idx, score in sorted(similar_user_posts.items(), key=lambda x: x[1], reverse=True)
    #             if self.user_item_matrix[user_id][post_idx] == 0  # Filter out posts user has already interacted with
    #         ][:n_recommendations]
            
    #         return recommendations
            
    #     except Exception as e:
    #         self.logger.error(f"Error in collaborative recommendations: {str(e)}")
    #         return self.get_popular_recommendations(n_recommendations)


    # def get_collaborative_recommendations(self, user_id: int, n_recommendations: int = 5) -> List[Dict]:
    #     """
    #     Get collaborative recommendations with proper index handling
    #     """
    #     try:
    #         # Get correct user index
    #         user_idx = self.get_user_matrix_index(user_id)
            
    #         # Calculate user similarity
    #         user_similarity = cosine_similarity([self.user_item_matrix[user_idx]], self.user_item_matrix)[0]
            
    #         # Get most similar users (excluding self)
    #         similar_users = np.argsort(user_similarity)[::-1]
    #         similar_users = [u for u in similar_users if u != user_idx][:5]
            
    #         if not similar_users:
    #             return self.get_popular_recommendations(n_recommendations)
            
    #         # Get their highly rated posts
    #         similar_user_posts = defaultdict(float)
            
    #         for sim_user_idx in similar_users:
    #             sim_score = user_similarity[sim_user_idx]
    #             user_ratings = self.user_item_matrix[sim_user_idx]
                
    #             for post_idx, rating in enumerate(user_ratings):
    #                 if rating > 0 and post_idx < len(self.post_features):
    #                     similar_user_posts[post_idx] += rating * sim_score
            
    #         # Sort and filter recommendations
    #         recommendations = [
    #             {
    #                 'post_id': int(self.post_features.iloc[post_idx]['post_id']),
    #                 'score': float(score),
    #                 'type': 'collaborative'
    #             }
    #             for post_idx, score in sorted(similar_user_posts.items(), key=lambda x: x[1], reverse=True)
    #             if self.user_item_matrix[user_idx][post_idx] == 0
    #         ][:n_recommendations]
            
    #         return recommendations
            
    #     except Exception as e:
    #         self.logger.error(f"Error in collaborative recommendations: {str(e)}")
    #         return self.get_popular_recommendations(n_recommendations)

# ************************************************


    # def get_collaborative_recommendations(self, user_id: int, n_recommendations: int = 5) -> List[Dict]:
    #     """
    #     Get collaborative recommendations with proper index handling and recency scoring
        
    #     Parameters:
    #     - user_id: ID of the user receiving recommendations
    #     - n_recommendations: Number of recommendations to return
        
    #     Returns:
    #     - List of dictionaries containing post recommendations with scores
    #     """
    #     try:
    #         # Get correct user index
    #         user_idx = self.get_user_matrix_index(user_id)
            
    #         # Calculate user similarity
    #         user_similarity = cosine_similarity([self.user_item_matrix[user_idx]], self.user_item_matrix)[0]
            
    #         # Get most similar users (excluding self)
    #         similar_users = np.argsort(user_similarity)[::-1]
    #         similar_users = [u for u in similar_users if u != user_idx][:5]
            
    #         if not similar_users:
    #             return self.get_popular_recommendations(n_recommendations)
            
    #         # Get post recommendations with combined similarity and recency scores
    #         similar_user_posts = defaultdict(float)
            
    #         for sim_user_idx in similar_users:
    #             sim_score = user_similarity[sim_user_idx]
    #             user_ratings = self.user_item_matrix[sim_user_idx]
                
    #             for post_idx, rating in enumerate(user_ratings):
    #                 if rating > 0 and post_idx < len(self.post_features):
    #                     # Get post creation date
    #                     post_date = pd.to_datetime(self.post_features.iloc[post_idx]['created_at'])
    #                     # print(f"this is type of post date {type(post_date)}")
                        
    #                     # Calculate recency score
    #                     recency_score = self.calculate_recency_score(post_date)
                        
    #                     # Combine similarity, rating, and recency scores
    #                     # Use weights to balance the influence of each factor
    #                     similarity_weight = 0.4
    #                     rating_weight = 0.3
    #                     recency_weight = 0.3
                        
    #                     combined_score = (
    #                         (sim_score * similarity_weight) +
    #                         (rating * rating_weight) +
    #                         (recency_score * recency_weight)
    #                     )
                        
    #                     similar_user_posts[post_idx] += combined_score
            
    #         # Sort and filter recommendations
    #         recommendations = []
    #         seen_posts = set()
            
    #         for post_idx, score in sorted(similar_user_posts.items(), key=lambda x: x[1], reverse=True):
    #             if len(recommendations) >= n_recommendations:
    #                 break
                    
    #             # Skip if user has already interacted with this post
    #             if self.user_item_matrix[user_idx][post_idx] > 0:
    #                 continue
                    
    #             # Skip if we've already recommended this post
    #             post_id = int(self.post_features.iloc[post_idx]['post_id'])
    #             if post_id in seen_posts:
    #                 continue
                    
    #             seen_posts.add(post_id)
    #             post_date = pd.to_datetime(self.post_features.iloc[post_idx]['created_at'])
    #             recency_score = self.calculate_recency_score(post_date)
                
    #             recommendations.append({
    #                 'post_id': post_id,
    #                 'score': float(score),
    #                 'type': 'collaborative',
    #                 'recency_score': float(recency_score)
    #             })
            
    #         return recommendations
            
    #     except Exception as e:
    #         self.logger.error(f"Error in collaborative recommendations: {str(e)}")
    #         return self.get_popular_recommendations(n_recommendations)


    #kok



    def get_collaborative_recommendations(self, user_id: int, n_recommendations: int = 5) -> List[Dict]:
        """
        Get collaborative recommendations with proper index handling and recency scoring
        """
        try:
            # Get correct user index
            user_idx = self.get_user_matrix_index(user_id)
            
            # Calculate user similarity
            user_similarity = cosine_similarity([self.user_item_matrix[user_idx]], self.user_item_matrix)[0]
            
            # Get most similar users (excluding self)
            similar_users = np.argsort(user_similarity)[::-1]
            similar_users = [u for u in similar_users if u != user_idx][:5]
            
            if not similar_users:
                return self.get_popular_recommendations(n_recommendations)
            
            # Get post recommendations with combined similarity and recency scores
            similar_user_posts = defaultdict(float)
            
            for sim_user_idx in similar_users:
                sim_score = user_similarity[sim_user_idx]
                user_ratings = self.user_item_matrix[sim_user_idx]
                
                for post_idx, rating in enumerate(user_ratings):
                    if rating > 0 and post_idx < len(self.post_features):
                        # Get post creation date
                        post_date = pd.to_datetime(self.post_features.iloc[post_idx]['created_at'])
                        
                        # Add debug logging
                        self.logger.debug(f"Post {post_idx} date: {post_date}")
                        
                        # Calculate recency score with proper date handling
                        if isinstance(post_date, str):
                            post_date = pd.to_datetime(post_date)
                        if post_date.tzinfo is not None:
                            post_date = post_date.tz_localize(None)
                            
                        recency_score = self.calculate_recency_score(post_date)
                        
                        # Add debug logging for recency calculation
                        self.logger.debug(f"Post {post_idx} recency score: {recency_score}")
                        
                        # Combine similarity, rating, and recency scores
                        similarity_weight = 0.4
                        rating_weight = 0.3
                        recency_weight = 0.3
                        
                        combined_score = (
                            (sim_score * similarity_weight) +
                            (rating * rating_weight) +
                            (recency_score * recency_weight)
                        )
                        
                        similar_user_posts[post_idx] += combined_score
            
            # Sort and filter recommendations
            recommendations = []
            seen_posts = set()
            
            # Add debug logging for final scores
            for post_idx, score in sorted(similar_user_posts.items(), key=lambda x: x[1], reverse=True):
                if len(recommendations) >= n_recommendations:
                    break
                    
                # Skip if user has already interacted with this post
                if self.user_item_matrix[user_idx][post_idx] > 0:
                    continue
                    
                # Skip if we've already recommended this post
                post_id = int(self.post_features.iloc[post_idx]['post_id'])
                if post_id in seen_posts:
                    continue
                    
                seen_posts.add(post_id)
                post_date = pd.to_datetime(self.post_features.iloc[post_idx]['created_at'])
                
                # Ensure proper date handling
                if isinstance(post_date, str):
                    post_date = pd.to_datetime(post_date)
                if post_date.tzinfo is not None:
                    post_date = post_date.tz_localize(None)
                    
                recency_score = self.calculate_recency_score(post_date)
                # Add this temporarily to check post ages
                
                
                # Debug log the post details
                self.logger.debug(f"Final recommendation - Post ID: {post_id}, Date: {post_date}, Recency Score: {recency_score}")
                
                recommendations.append({
                    'post_id': post_id,
                    'score': float(score),
                    'type': 'collaborative',
                    'recency_score': float(recency_score)
                })
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error in collaborative recommendations: {str(e)}")
            return self.get_popular_recommendations(n_recommendations)
        




    def get_popular_recommendations(self, n_recommendations: int = 5) -> List[Dict]:
        """
        Get popular posts with recency factor
        """
        try:
            # Calculate base popularity scores
            popularity_scores = np.sum(self.user_item_matrix, axis=0)
            
            # Calculate recency scores
            recency_scores = np.array([
                self.calculate_recency_score(
                    pd.to_datetime(self.post_features.iloc[i]['created_at'])
                )
                for i in range(len(popularity_scores))
            ])
            
            # Combine popularity and recency (with weights)
            popularity_weight = 0.7
            recency_weight = 0.3
            
            # Normalize popularity scores
            normalized_popularity = (popularity_scores - popularity_scores.min()) / \
                                 (popularity_scores.max() - popularity_scores.min())
            
            final_scores = (normalized_popularity * popularity_weight) + \
                          (recency_scores * recency_weight)
            
            # Get top posts
            top_posts = np.argsort(final_scores)[::-1][:n_recommendations]
            
            return [
                {
                    'post_id': int(self.post_features.iloc[i]['post_id']),
                    'score': float(final_scores[i]),
                    'type': 'popular',
                    'recency_score': float(recency_scores[i]),
                    'popularity_score': float(normalized_popularity[i])
                }
                for i in top_posts
            ]
            
        except Exception as e:
            self.logger.error(f"Error getting popular recommendations: {str(e)}")
            return []


    

    # def get_hybrid_recommendations(self, user_id: int, n_recommendations: int = 5, 
    #                              connection_weight: float = 0.8) -> List[Dict]:
    #     """
    #     Get hybrid recommendations prioritizing posts from connections
    #     """
    #     try:
    #          # Get user's connections
    #         user_connections = set(self.user_connections.get(user_id, []))
            
    #         # Get base recommendations
    #         content_recs = self.get_content_based_recommendations(user_id, n_recommendations * 2)
    #         collab_recs = self.get_collaborative_recommendations(user_id, n_recommendations * 2)
            
    #         # Combine and normalize scores
    #         all_recs = defaultdict(lambda: {'score': 0, 'count': 0, 'types': set(), 'from_connection': False})
            
    #         for rec in content_recs + collab_recs:
    #             post_id = rec['post_id']
    #             post_author = int(self.post_features[
    #                 self.post_features['post_id'] == post_id
    #             ]['user_id'].iloc[0])
                
    #             # Check if post is from a connection
    #             is_from_connection = post_author in user_connections
                
    #             # Apply connection boost to score
    #             score = rec['score']
    #             if is_from_connection:
    #                 score *= (1 + connection_weight)
                
    #             all_recs[post_id]['score'] += score
    #             all_recs[post_id]['count'] += 1
    #             all_recs[post_id]['types'].add(rec['type'])
    #             all_recs[post_id]['from_connection'] = is_from_connection
            
    #         # Calculate final scores and sort
    #         final_recommendations = [
    #             {
    #                 'post_id': post_id,
    #                 'score': info['score'] / info['count'],
    #                 'types': list(info['types']),
    #                 'from_connection': info['from_connection']
    #             }
    #             for post_id, info in all_recs.items()
    #         ]
            
    #         # Sort by connection status first, then by score
    #         final_recommendations.sort(
    #             key=lambda x: (x['from_connection'], x['score']), 
    #             reverse=True
    #         )
            
    #         return final_recommendations[:n_recommendations]
            
    #     except Exception as e:
    #         self.logger.error(f"Error in hybrid recommendations: {str(e)}")
    #         return self.get_popular_recommendations(n_recommendations)


    # def get_hybrid_recommendations(self, user_id: int, n_recommendations: int = 5, 
    #                          connection_weight: float = 0.8, connection_ratio: float = 0.7) -> List[Dict]:
    #     """
    #     Get hybrid recommendations with controlled ratio of posts from connections
        
    #     Parameters:
    #     - user_id: ID of the user receiving recommendations
    #     - n_recommendations: Total number of recommendations to return
    #     - connection_weight: Weight multiplier for connection posts' scores
    #     - connection_ratio: Target ratio of recommendations that should come from connections (0.0 to 1.0)
    #     """
    #     try:
    #         # Get user's connections
    #         user_connections = set(self.user_connections.get(user_id, []))
            
    #         # Get base recommendations
    #         content_recs = self.get_content_based_recommendations(user_id, n_recommendations * 2)
    #         collab_recs = self.get_collaborative_recommendations(user_id, n_recommendations * 2)
            
    #         # Combine and normalize scores
    #         connection_recs = []
    #         non_connection_recs = []
            
    #         # Process all recommendations
    #         for rec in content_recs + collab_recs:
    #             post_id = rec['post_id']
    #             post_author = int(self.post_features[
    #                 self.post_features['post_id'] == post_id
    #             ]['user_id'].iloc[0])
                
    #             # Check if post is from a connection
    #             is_from_connection = post_author in user_connections
                
    #             # Apply connection boost to score
    #             score = rec['score']
    #             if is_from_connection:
    #                 score *= (1 + connection_weight)
                
    #             recommendation = {
    #                 'post_id': post_id,
    #                 'score': score,
    #                 'types': {rec['type']},
    #                 'from_connection': is_from_connection,
    #                 'author_id': post_author
    #             }
                
    #             # Separate into connection and non-connection recommendations
    #             if is_from_connection:
    #                 connection_recs.append(recommendation)
    #             else:
    #                 non_connection_recs.append(recommendation)
            
    #         # Remove duplicates and sort both lists by score
    #         connection_recs = self._deduplicate_and_sort(connection_recs)
    #         non_connection_recs = self._deduplicate_and_sort(non_connection_recs)
            
    #         # Calculate how many recommendations should come from connections
    #         target_connection_count = int(n_recommendations * connection_ratio)
            
    #         # Adjust if we don't have enough connection recommendations
    #         actual_connection_count = min(target_connection_count, len(connection_recs))
    #         non_connection_count = n_recommendations - actual_connection_count
            
    #         # Combine the recommendations
    #         final_recommendations = (
    #             connection_recs[:actual_connection_count] +
    #             non_connection_recs[:non_connection_count]
    #         )
            
    #         # Sort final recommendations by score
    #         final_recommendations.sort(key=lambda x: x['score'], reverse=True)
            
    #         # Format the output
    #         return [
    #             {
    #                 'post_id': rec['post_id'],
    #                 'score': rec['score'],
    #                 'types': list(rec['types']),
    #                 'from_connection': rec['from_connection'],
    #                 'author_id': rec['author_id']
    #             }
    #             for rec in final_recommendations
    #         ]
                
    #     except Exception as e:
    #         self.logger.error(f"Error in hybrid recommendations: {str(e)}")
    #         return self.get_popular_recommendations(n_recommendations)



    # def get_hybrid_recommendations(self, user_id: int, n_recommendations: int = 5, 
    #                          connection_weight: float = 0.8, connection_ratio: float = 0.7) -> List[Dict]:
    #     """
    #     Get hybrid recommendations with controlled ratio of posts from connections
    #     """
    #     try:
    #         # Get user's connections as a list of integers
    #         user_connections = set(self.get_user_connections(user_id))
    #         print(f"these are user connections{user_connections}")
            
    #         # Rest of the method remains the same...
    #         content_recs = self.get_content_based_recommendations(user_id, n_recommendations * 2)
    #         collab_recs = self.get_collaborative_recommendations(user_id, n_recommendations * 2)
            
    #         # Combine and normalize scores
    #         connection_recs = []
    #         non_connection_recs = []
    #         seen_posts = set()
            
    #         # Process all recommendations
    #         for rec in content_recs + collab_recs:
    #             post_id = rec['post_id']
                
    #             # Skip if we've already processed this post
    #             if post_id in seen_posts:
    #                 continue
    #             seen_posts.add(post_id)
                
    #             try:
    #                 post_data = self.post_features[self.post_features['post_id'] == post_id].iloc[0]
    #                 post_author = int(post_data['user_id'])
    #             except (IndexError, KeyError):
    #                 continue
                
    #             # Check if post is from a connection
    #             is_from_connection = post_author in user_connections
                
    #             # Apply connection boost to score
    #             score = rec['score']
    #             if is_from_connection:
    #                 score *= (1 + connection_weight)
                
    #             recommendation = {
    #                 'post_id': post_id,
    #                 'score': score,
    #                 'types': {rec['type']},
    #                 'from_connection': is_from_connection,
    #                 'author_id': post_author
    #             }
                
    #             # Separate into connection and non-connection recommendations
    #             if is_from_connection:
    #                 connection_recs.append(recommendation)
    #             else:
    #                 non_connection_recs.append(recommendation)
            
    #         # Sort both lists by score
    #         connection_recs.sort(key=lambda x: x['score'], reverse=True)
    #         non_connection_recs.sort(key=lambda x: x['score'], reverse=True)
            
    #         # Calculate how many recommendations should come from connections
    #         target_connection_count = int(n_recommendations * connection_ratio)
            
    #         # Adjust if we don't have enough connection recommendations
    #         actual_connection_count = min(target_connection_count, len(connection_recs))
    #         non_connection_count = n_recommendations - actual_connection_count
            
    #         # Combine the recommendations
    #         final_recommendations = (
    #             connection_recs[:actual_connection_count] +
    #             non_connection_recs[:non_connection_count]
    #         )
            
    #         # Sort final recommendations by score
    #         final_recommendations.sort(key=lambda x: x['score'], reverse=True)
            
    #         # Format the output
    #         return [
    #             {
    #                 'post_id': rec['post_id'],
    #                 'score': rec['score'],
    #                 'types': list(rec['types']),
    #                 'from_connection': rec['from_connection'],
    #                 'author_id': rec['author_id']
    #             }
    #             for rec in final_recommendations
    #         ]
                
    #     except Exception as e:
    #         self.logger.error(f"Error in hybrid recommendations: {str(e)}")
    #         return self.get_popular_recommendations(n_recommendations)
#***********************************************************************************************************************************************


    # def get_hybrid_recommendations(self, user_id: int, n_recommendations: int = 5, 
    #                          connection_weight: float = 2.0, connection_ratio: float = 0.8) -> List[Dict]:
    #     """
    #     Get hybrid recommendations prioritizing posts from user's connections
    #     """
    #     try:
    #         # Get user's connections from the stored data
    #         user_connections = set(self.user_connections.get(user_id, []))
            
    #         # Log the connections for debugging
    #         self.logger.info(f"Found connections for user {user_id}: {user_connections}")
            
    #         # Get base recommendations
    #         content_recs = self.get_content_based_recommendations(user_id, n_recommendations * 3)
    #         collab_recs = self.get_collaborative_recommendations(user_id, n_recommendations * 3)
            
    #         # Separate recommendations by connection status
    #         connection_recs = []
    #         non_connection_recs = []
    #         seen_posts = set()
            
    #         # Process all recommendations
    #         for rec in content_recs + collab_recs:
    #             post_id = rec['post_id']
                
    #             if post_id in seen_posts:
    #                 continue
    #             seen_posts.add(post_id)
                
    #             try:
    #                 post_data = self.post_features[self.post_features['post_id'] == post_id].iloc[0]
    #                 post_author = int(post_data['user_id'])
                    
    #                 # Check if post is from a connection
    #                 is_from_connection = post_author in user_connections
                    
    #                 # Calculate score with higher weight for connection posts
    #                 score = rec['score']
    #                 if is_from_connection:
    #                     score *= connection_weight
                    
    #                 recommendation = {
    #                     'post_id': post_id,
    #                     'score': score,
    #                     'types': {rec['type']},
    #                     'from_connection': is_from_connection,
    #                     'author_id': post_author
    #                 }
                    
    #                 if is_from_connection:
    #                     connection_recs.append(recommendation)
    #                 else:
    #                     non_connection_recs.append(recommendation)
                        
    #             except (IndexError, KeyError) as e:
    #                 continue
            
    #         # Log recommendation counts for debugging
    #         self.logger.info(f"Found {len(connection_recs)} connection recommendations and {len(non_connection_recs)} non-connection recommendations")
            
    #         # Sort both lists by score
    #         connection_recs.sort(key=lambda x: x['score'], reverse=True)
    #         non_connection_recs.sort(key=lambda x: x['score'], reverse=True)
            
    #         # Calculate target number of connection recommendations
    #         target_connection_count = int(n_recommendations * connection_ratio)
    #         actual_connection_count = min(target_connection_count, len(connection_recs))
    #         non_connection_count = n_recommendations - actual_connection_count
            
    #         # Combine recommendations
    #         final_recommendations = (
    #             connection_recs[:actual_connection_count] +
    #             non_connection_recs[:non_connection_count]
    #         )
            
    #         # Sort final recommendations by score
    #         final_recommendations.sort(key=lambda x: x['score'], reverse=True)
            
    #         return [
    #             {
    #                 'post_id': rec['post_id'],
    #                 'score': rec['score'],
    #                 'types': list(rec['types']),
    #                 'from_connection': rec['from_connection'],
    #                 'author_id': rec['author_id']
    #             }
    #             for rec in final_recommendations
    #         ]
                
    #     except Exception as e:
    #         self.logger.error(f"Error in hybrid recommendations: {str(e)}")
    #         return self.get_popular_recommendations(n_recommendations)


#this is onlly from connections
    # def get_hybrid_recommendations(self, user_id: int, n_recommendations: int = 5) -> List[Dict]:
    #     """
    #     Get recommendations only from user's connections, ignoring content-based and collaborative filtering
    #     """
    #     try:
    #         # Get user's connections from the stored data
    #         user_connections = set(self.user_connections.get(user_id, []))
            
    #         # Log the connections for debugging
    #         self.logger.info(f"Finding posts from connections for user {user_id}: {user_connections}")
            
    #         if not user_connections:
    #             self.logger.info(f"No connections found for user {user_id}")
    #             return self.get_popular_recommendations(n_recommendations)
            
    #         # Get all posts from connections
    #         connection_posts = []
    #         for idx, post in self.post_features.iterrows():
    #             try:
    #                 post_author = int(post['user_id'])
    #                 if post_author in user_connections:
    #                     # Calculate a simple recency score for ranking
    #                     recency_score = self.calculate_recency_score(post['created_at'])
                        
    #                     connection_posts.append({
    #                         'post_id': int(post['post_id']),
    #                         'score': recency_score,  # Using recency as the primary score
    #                         'types': ['connection'],
    #                         'from_connection': True,
    #                         'author_id': post_author
    #                     })
    #             except (ValueError, KeyError) as e:
    #                 continue
            
    #         # Sort by recency score
    #         connection_posts.sort(key=lambda x: x['score'], reverse=True)
            
    #         # Return the top N recommendations
    #         return connection_posts[:n_recommendations]
                
    #     except Exception as e:
    #         self.logger.error(f"Error in connection-only recommendations: {str(e)}")
    #         return self.get_popular_recommendations(n_recommendations)


#this is perfect for connection basesd first and after that collabarative and contetnt based and popular post

    # def get_hybrid_recommendations(self, user_id: int, n_recommendations: int = 5) -> List[Dict]:
    #     """
    #     Get recommendations prioritizing posts from connections, with fallback to other recommendation types
    #     when there aren't enough connection posts
    #     """
    #     try:
    #         # Get user's connections from the stored data
    #         user_connections = set(self.user_connections.get(user_id, []))
            
    #         # Log the connections for debugging
    #         self.logger.info(f"Finding posts from connections for user {user_id}: {user_connections}")
            
    #         # Initialize recommendations list
    #         recommendations = []
            
    #         # First try to get posts from connections
    #         if user_connections:
    #             for idx, post in self.post_features.iterrows():
    #                 try:
    #                     post_author = int(post['user_id'])
    #                     if post_author in user_connections:
    #                         recency_score = self.calculate_recency_score(post['created_at'])
                            
    #                         recommendations.append({
    #                             'post_id': int(post['post_id']),
    #                             'score': recency_score,
    #                             'types': ['connection'],
    #                             'from_connection': True,
    #                             'author_id': post_author
    #                         })
    #                 except (ValueError, KeyError) as e:
    #                     continue
                
    #             # Sort connection recommendations by recency
    #             recommendations.sort(key=lambda x: x['score'], reverse=True)
            
    #         # Check if we need more recommendations
    #         remaining_recommendations = n_recommendations - len(recommendations)
            
    #         if remaining_recommendations > 0:
    #             self.logger.info(f"Need {remaining_recommendations} more recommendations. Using fallback methods.")
                
    #             # Track posts we've already recommended
    #             recommended_post_ids = {rec['post_id'] for rec in recommendations}
                
    #             # Get additional recommendations from collaborative filtering
    #             try:
    #                 collab_recs = self.get_collaborative_recommendations(user_id, remaining_recommendations * 2)
    #                 for rec in collab_recs:
    #                     if (len(recommendations) < n_recommendations and 
    #                         rec['post_id'] not in recommended_post_ids):
    #                         rec['types'] = ['collaborative']
    #                         rec['from_connection'] = False
    #                         recommendations.append(rec)
    #                         recommended_post_ids.add(rec['post_id'])
    #             except Exception as e:
    #                 self.logger.error(f"Error getting collaborative recommendations: {str(e)}")
                
    #             # If we still need more, try content-based
    #             remaining_recommendations = n_recommendations - len(recommendations)
    #             if remaining_recommendations > 0:
    #                 try:
    #                     content_recs = self.get_content_based_recommendations(user_id, remaining_recommendations * 2)
    #                     for rec in content_recs:
    #                         if (len(recommendations) < n_recommendations and 
    #                             rec['post_id'] not in recommended_post_ids):
    #                             rec['types'] = ['content']
    #                             rec['from_connection'] = False
    #                             recommendations.append(rec)
    #                             recommended_post_ids.add(rec['post_id'])
    #                 except Exception as e:
    #                     self.logger.error(f"Error getting content-based recommendations: {str(e)}")
                
    #             # If we still need more, fall back to popular recommendations
    #             remaining_recommendations = n_recommendations - len(recommendations)
    #             if remaining_recommendations > 0:
    #                 try:
    #                     popular_recs = self.get_popular_recommendations(remaining_recommendations)
    #                     for rec in popular_recs:
    #                         if (len(recommendations) < n_recommendations and 
    #                             rec['post_id'] not in recommended_post_ids):
    #                             rec['types'] = ['popular']
    #                             rec['from_connection'] = False
    #                             recommendations.append(rec)
    #                             recommended_post_ids.add(rec['post_id'])
    #                 except Exception as e:
    #                     self.logger.error(f"Error getting popular recommendations: {str(e)}")
            
    #         # Final sort of all recommendations (prioritizing connection posts)
    #         recommendations.sort(key=lambda x: (x['from_connection'], x['score']), reverse=True)
            
    #         return recommendations[:n_recommendations]
                
    #     except Exception as e:
    #         self.logger.error(f"Error in hybrid recommendations: {str(e)}")
    #         return self.get_popular_recommendations(n_recommendations)

# ********************************************************************new latest***************************

    # def get_hybrid_recommendations(self, user_id: int, n_recommendations: int = 5,
    #                          connection_ratio: Optional[float] = None,
    #                          max_post_age_days: Optional[int] = None) -> List[Dict]:
    #     """
    #     Get recommendations with configurable ratio between connection and non-connection posts
        
    #     Parameters:
    #     - user_id: ID of the user receiving recommendations
    #     - n_recommendations: Total number of recommendations to return
    #     - connection_ratio: Optional; if provided, controls the ratio of connection posts (0.0 to 1.0)
    #                     If None, returns all recent connection posts first
    #     - max_post_age_days: Optional; maximum age of connection posts in days
    #                         If None, no time restriction is applied
    #     """
    #     try:
    #         # Get user's connections
    #         user_connections = set(self.user_connections.get(user_id, []))
    #         self.logger.info(f"Finding posts from connections for user {user_id}: {user_connections}")
            
    #         # Get current time for age comparison
    #         current_time = pd.Timestamp.now()
            
    #         # Initialize recommendations lists
    #         connection_recommendations = []
    #         non_connection_recommendations = []
    #         recommended_post_ids = set()
            
    #         # First gather all connection posts that meet the age criterion
    #         if user_connections:
    #             for idx, post in self.post_features.iterrows():
    #                 try:
    #                     post_author = int(post['user_id'])
    #                     post_date = pd.to_datetime(post['created_at'])
    #                     if post_author in user_connections:
    #                         if isinstance(post_date,str):
    #                             post_date=pd.to_datetime(post_date).replace(tzinfo=None)
    #                         elif hasattr(post_date,'tzinfo') and post_date.tzinfo is not None:
    #                             post_date=post_date.replace(tzinfo=None)
    #                         # post_date = pd.to_datetime(post['created_at'])
    #                         post_age_days = (current_time - post_date).days
                            
    #                         # Check if post meets age criterion
    #                         if max_post_age_days is not None and post_age_days > max_post_age_days:
    #                             continue
                                
    #                         recency_score = self.calculate_recency_score(post['created_at'])
                            
    #                         connection_recommendations.append({
    #                             'post_id': int(post['post_id']),
    #                             'score': recency_score,
    #                             'types': ['connection'],
    #                             'from_connection': True,
    #                             'author_id': post_author,
    #                             'age_days': post_age_days
    #                         })
    #                         recommended_post_ids.add(int(post['post_id']))
                            
    #                 except (ValueError, KeyError) as e:
    #                     continue
                
    #             # Sort connection recommendations by recency
    #             connection_recommendations.sort(key=lambda x: x['score'], reverse=True)
            
    #         # Calculate how many recommendations we need from each source
    #         if connection_ratio is not None:
    #             target_connection_count = int(n_recommendations * connection_ratio)
    #             target_non_connection_count = n_recommendations - target_connection_count
    #         else:
    #             # If no ratio specified, try to fill with connection posts first
    #             target_connection_count = len(connection_recommendations)
    #             target_non_connection_count = n_recommendations - target_connection_count
            
    #         # If we need non-connection recommendations, gather them
    #         if target_non_connection_count > 0:
    #             # Try collaborative filtering first
    #             try:
    #                 collab_recs = self.get_collaborative_recommendations(user_id, target_non_connection_count * 2)
    #                 for rec in collab_recs:
    #                     if rec['post_id'] not in recommended_post_ids:
    #                         rec['types'] = ['collaborative']
    #                         rec['from_connection'] = False
    #                         non_connection_recommendations.append(rec)
    #                         recommended_post_ids.add(rec['post_id'])
    #             except Exception as e:
    #                 self.logger.error(f"Error getting collaborative recommendations: {str(e)}")
                
    #             # If we still need more, try content-based
    #             if len(non_connection_recommendations) < target_non_connection_count:
    #                 remaining = target_non_connection_count - len(non_connection_recommendations)
    #                 try:
    #                     content_recs = self.get_content_based_recommendations(user_id, remaining * 2)
    #                     for rec in content_recs:
    #                         if rec['post_id'] not in recommended_post_ids:
    #                             rec['types'] = ['content']
    #                             rec['from_connection'] = False
    #                             non_connection_recommendations.append(rec)
    #                             recommended_post_ids.add(rec['post_id'])
    #                 except Exception as e:
    #                     self.logger.error(f"Error getting content-based recommendations: {str(e)}")
                
    #             # Sort non-connection recommendations by score
    #             non_connection_recommendations.sort(key=lambda x: x['score'], reverse=True)
            
    #         # Combine recommendations according to target counts
    #         final_recommendations = (
    #             connection_recommendations[:target_connection_count] +
    #             non_connection_recommendations[:target_non_connection_count]
    #         )
            
    #         # If we still don't have enough recommendations, fall back to popular posts
    #         if len(final_recommendations) < n_recommendations:
    #             remaining = n_recommendations - len(final_recommendations)
    #             try:
    #                 popular_recs = self.get_popular_recommendations(remaining)
    #                 for rec in popular_recs:
    #                     if rec['post_id'] not in recommended_post_ids:
    #                         rec['types'] = ['popular']
    #                         rec['from_connection'] = False
    #                         final_recommendations.append(rec)
    #             except Exception as e:
    #                 self.logger.error(f"Error getting popular recommendations: {str(e)}")
            
    #         # Final sort ensuring connection posts are prioritized
    #         final_recommendations.sort(key=lambda x: (x['from_connection'], x['score']), reverse=True)
            
    #         return final_recommendations[:n_recommendations]
                
    #     except Exception as e:
    #         self.logger.error(f"Error in hybrid recommendations: {str(e)}")
    #         return self.get_popular_recommendations(n_recommendations)



    def get_hybrid_recommendations(self, user_id: int, n_recommendations: int = 5,
                         connection_ratio: Optional[float] = None,
                         max_post_age_days: Optional[int] = None,
                         detailed_response: bool = False) -> Union[List[int], List[Dict]]:
        """
        Get recommendations with configurable ratio between connection and non-connection posts
        
        Parameters:
        - user_id: ID of the user receiving recommendations
        - n_recommendations: Total number of recommendations to return
        - connection_ratio: Optional; if provided, controls the ratio of connection posts (0.0 to 1.0)
                        If None, returns all recent connection posts first
        - max_post_age_days: Optional; maximum age of connection posts in days
                            If None, no time restriction is applied
        - detailed_response: If True, returns full recommendation details; if False, returns just post IDs
        
        Returns:
        - If detailed_response is False: List[int] containing just post IDs
        - If detailed_response is True: List[Dict] containing full recommendation details
        """
        try:
            # Get user's connections
            user_connections = set(self.user_connections.get(user_id, []))
            self.logger.info(f"Finding posts from connections for user {user_id}: {user_connections}")
            
            # Get current time for age comparison
            current_time = pd.Timestamp.now()
            
            # Initialize recommendations lists
            connection_recommendations = []
            non_connection_recommendations = []
            recommended_post_ids = set()
            
            # First gather all connection posts that meet the age criterion
            if user_connections:
                for idx, post in self.post_features.iterrows():
                    try:
                        post_author = int(post['user_id'])
                        post_date = pd.to_datetime(post['created_at'])
                        if post_author in user_connections:
                            if isinstance(post_date, str):
                                post_date = pd.to_datetime(post_date).replace(tzinfo=None)
                            elif hasattr(post_date, 'tzinfo') and post_date.tzinfo is not None:
                                post_date = post_date.replace(tzinfo=None)
                                
                            post_age_days = (current_time - post_date).days
                            
                            # Check if post meets age criterion
                            if max_post_age_days is not None and post_age_days > max_post_age_days:
                                continue
                                
                            recency_score = self.calculate_recency_score(post['created_at'])
                            print(f"this is hybrid recency type{type(recency_score)}")
                            
                            if detailed_response:
                                rec = {
                                    'post_id': int(post['post_id']),
                                    'score': recency_score,
                                    'types': ['connection'],
                                    'from_connection': True,
                                    'author_id': post_author,
                                    'age_days': post_age_days
                                }
                            else:
                                rec = int(post['post_id'])
                                
                            connection_recommendations.append(rec)
                            recommended_post_ids.add(int(post['post_id']))
                            
                    except (ValueError, KeyError) as e:
                        continue
                
                # Sort connection recommendations
                if detailed_response:
                    connection_recommendations.sort(key=lambda x: x['score'], reverse=True)
            
            # Calculate how many recommendations we need from each source
            if connection_ratio is not None:
                target_connection_count = int(n_recommendations * connection_ratio)
                target_non_connection_count = n_recommendations - target_connection_count
            else:
                target_connection_count = len(connection_recommendations)
                target_non_connection_count = n_recommendations - target_connection_count
            
            # If we need non-connection recommendations, gather them
            if target_non_connection_count > 0:
                # Try collaborative filtering first
                try:
                    collab_recs = self.get_collaborative_recommendations(user_id, target_non_connection_count * 2)
                    for rec in collab_recs:
                        post_id = rec['post_id']
                        if post_id not in recommended_post_ids:
                            if detailed_response:
                                rec['types'] = ['collaborative']
                                rec['from_connection'] = False
                                non_connection_recommendations.append(rec)
                            else:
                                non_connection_recommendations.append(post_id)
                            recommended_post_ids.add(post_id)
                except Exception as e:
                    self.logger.error(f"Error getting collaborative recommendations: {str(e)}")
                
                # Try content-based if needed
                if len(non_connection_recommendations) < target_non_connection_count:
                    remaining = target_non_connection_count - len(non_connection_recommendations)
                    try:
                        content_recs = self.get_content_based_recommendations(user_id, remaining * 2)
                        for rec in content_recs:
                            post_id = rec['post_id']
                            if post_id not in recommended_post_ids:
                                if detailed_response:
                                    rec['types'] = ['content']
                                    rec['from_connection'] = False
                                    non_connection_recommendations.append(rec)
                                else:
                                    non_connection_recommendations.append(post_id)
                                recommended_post_ids.add(post_id)
                    except Exception as e:
                        self.logger.error(f"Error getting content-based recommendations: {str(e)}")
                
                # Sort non-connection recommendations if detailed
                if detailed_response:
                    non_connection_recommendations.sort(key=lambda x: x['score'], reverse=True)
            
            # Combine recommendations according to target counts
            final_recommendations = (
                connection_recommendations[:target_connection_count] +
                non_connection_recommendations[:target_non_connection_count]
            )
            
            # Add popular recommendations if needed
            if len(final_recommendations) < n_recommendations:
                remaining = n_recommendations - len(final_recommendations)
                try:
                    popular_recs = self.get_popular_recommendations(remaining)
                    for rec in popular_recs:
                        post_id = rec['post_id'] if detailed_response else rec
                        if post_id not in recommended_post_ids:
                            if detailed_response:
                                rec['types'] = ['popular']
                                rec['from_connection'] = False
                                final_recommendations.append(rec)
                            else:
                                final_recommendations.append(post_id)
                            recommended_post_ids.add(post_id)
                except Exception as e:
                    self.logger.error(f"Error getting popular recommendations: {str(e)}")
            
            # Final sort if detailed
            if detailed_response:
                final_recommendations.sort(key=lambda x: (x['from_connection'], x['score']), reverse=True)
            
            return final_recommendations[:n_recommendations]
                
        except Exception as e:
            self.logger.error(f"Error in hybrid recommendations: {str(e)}")
            fallback_recs = self.get_popular_recommendations(n_recommendations)
            if not detailed_response:
                return [rec['post_id'] for rec in fallback_recs]
            return fallback_recs


