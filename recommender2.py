import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import redis
import json
from datetime import datetime
from typing import List, Dict, Tuple
import logging


class TravelRecommender:
    def __init__(self, redis_host='localhost', redis_port=6379):
        self.redis_client = redis.Redis(host=redis_host, port=redis_port)
        self.cache_ttl = 3600
        
        # Initialize storage for our models
        self.post_vectors = None
        self.user_item_matrix = None
        self.post_features = None
        self.similarity_matrix = None
        
        # Add ID mapping dictionaries
        self.user_id_to_idx = {}
        self.idx_to_user_id = {}
        self.post_id_to_idx = {}
        self.idx_to_post_id = {}
        
        # Initialize logger
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        self.is_initialized = False

    def process_post_content(self, posts_df: pd.DataFrame) -> np.ndarray:
        """
        Process post content using TF-IDF vectorization
        """
        if posts_df.empty:
            raise ValueError("Posts DataFrame is empty")
            
        # Ensure required columns exist
        required_columns = ['title', 'description', 'location', 'tags']
        missing_columns = [col for col in required_columns if col not in posts_df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Handle missing values
        posts_df['title'] = posts_df['title'].fillna('')
        posts_df['description'] = posts_df['description'].fillna('')
        posts_df['location'] = posts_df['location'].fillna('')
        posts_df['tags'] = posts_df['tags'].fillna('')
        
        # Combine relevant text fields
        posts_df['combined_text'] = posts_df.apply(
            lambda x: f"{x['title']} {x['description']} {x['location']} {x['tags']}", 
            axis=1
        )
        
        # Create TF-IDF vectors
        vectorizer = TfidfVectorizer(
            max_features=100,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        try:
            vectors = vectorizer.fit_transform(posts_df['combined_text'])
            return vectors
        except Exception as e:
            self.logger.error(f"Error in TF-IDF vectorization: {str(e)}")
            raise

    def build_user_item_matrix(self, interactions_df: pd.DataFrame) -> np.ndarray:
        """
        Build user-item interaction matrix with weights
        """
        # Define interaction weights
        weights = {
            'view': 1,
            'like': 3,
            'comment': 4,
            'share': 5,
            'save':3
        }
        
        # Create pivot table with weighted interactions
        matrix = pd.pivot_table(
            interactions_df,
            values='interaction_type',
            index='user_id',
            columns='post_id',
            aggfunc=lambda x: sum(weights.get(i, 1) for i in x),
            fill_value=0
        )
        
        return matrix.values


    def create_id_mappings(self, posts_df: pd.DataFrame, interactions_df: pd.DataFrame):
        """
        Create mappings between actual IDs and array indices
        """
        if posts_df.empty or interactions_df.empty:
            raise ValueError("Either posts or interactions DataFrame is empty")
            
        if 'post_id' not in posts_df.columns or 'user_id' not in interactions_df.columns:
            raise ValueError("Required ID columns missing")
        
        # Create post ID mappings
        unique_post_ids = sorted(posts_df['post_id'].unique())
        self.post_id_to_idx = {pid: idx for idx, pid in enumerate(unique_post_ids)}
        self.idx_to_post_id = {idx: pid for pid, idx in self.post_id_to_idx.items()}
        
        # Create user ID mappings
        unique_user_ids = sorted(interactions_df['user_id'].unique())
        self.user_id_to_idx = {uid: idx for idx, uid in enumerate(unique_user_ids)}
        self.idx_to_user_id = {idx: uid for uid, idx in self.user_id_to_idx.items()}
        
        self.logger.info(f"Created mappings for {len(unique_post_ids)} posts and {len(unique_user_ids)} users")

    def update_models(self, posts_df: pd.DataFrame, interactions_df: pd.DataFrame):
        """
        Update all models with new data
        """
        try:
            self.logger.info("Starting model update...")
            
            # Validate input data
            if posts_df.empty or interactions_df.empty:
                raise ValueError("Empty DataFrame provided")
            
            # Create ID mappings first
            self.create_id_mappings(posts_df, interactions_df)
            
            # Store post features
            self.post_features = posts_df.copy()
            
            # Update content-based features
            self.post_vectors = self.process_post_content(posts_df)
            
            # Update collaborative filtering matrix
            self.user_item_matrix = self.build_user_item_matrix(interactions_df)
            
            # Calculate post similarity matrix
            self.similarity_matrix = cosine_similarity(self.post_vectors)
            
            # Set initialization flag
            self.is_initialized = True
            
            self.logger.info("Model update completed successfully")
            
        except Exception as e:
            self.logger.error(f"Error updating models: {str(e)}")
            self.is_initialized = False
            raise

    def check_initialization(self):
        """
        Check if the model is properly initialized
        """
        if not self.is_initialized:
            raise RuntimeError("Recommender not initialized. Please call update_models first.")
        
        if self.post_vectors is None or self.user_item_matrix is None or self.post_features is None:
            raise RuntimeError("Model components not properly initialized")

    def get_content_based_recommendations(self, user_id: int, n_recommendations: int = 5) -> List[Dict]:
        """
        Get content-based recommendations based on user's interaction history
        """
        try:
            self.check_initialization()
            
            # Map user_id to index
            user_idx = self.user_id_to_idx.get(user_id)
            if user_idx is None:
                self.logger.info(f"Unknown user_id: {user_id}, falling back to popular recommendations")
                return self.get_popular_recommendations(n_recommendations)

            # Get user's interaction history
            user_interactions = self.user_item_matrix[user_idx]
            
            # Find posts user has interacted with
            interacted_posts = np.where(user_interactions > 0)[0]
            
            if len(interacted_posts) == 0:
                self.logger.info(f"No interactions found for user_id: {user_id}")
                return self.get_popular_recommendations(n_recommendations)
            
            # Calculate average similarity with interacted posts
            sim_scores = np.mean([self.similarity_matrix[i] for i in interacted_posts], axis=0)
            
            # Get top similar posts
            similar_posts = np.argsort(sim_scores)[::-1]
            
            # Filter out already interacted posts and map back to real post IDs
            recommendations = [
                {
                    'post_id': self.idx_to_post_id[i],
                    'score': float(sim_scores[i]),
                    'type': 'content'
                }
                for i in similar_posts
                if i not in interacted_posts and i in self.idx_to_post_id
            ][:n_recommendations]
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error in content-based recommendations: {str(e)}")
            return self.get_popular_recommendations(n_recommendations)

    # ... [rest of the methods remain the same] ...
    def get_collaborative_recommendations(self, user_id: int, n_recommendations: int = 5) -> List[Dict]:
        """
        Get collaborative filtering recommendations using user similarity
        """
        try:
            # Calculate user similarity
            user_similarity = cosine_similarity([self.user_item_matrix[user_id]], self.user_item_matrix)[0]
            
            # Get most similar users
            similar_users = np.argsort(user_similarity)[::-1][1:6]  # Get top 5 similar users
            
            # Get their highly rated posts
            similar_user_posts = defaultdict(float)
            
            for sim_user_idx in similar_users:
                sim_score = user_similarity[sim_user_idx]
                user_ratings = self.user_item_matrix[sim_user_idx]
                
                for post_idx, rating in enumerate(user_ratings):
                    if rating > 0:
                        similar_user_posts[post_idx] += rating * sim_score
            
            # Sort and filter recommendations
            recommendations = [
                {
                    'post_id': int(self.post_features.iloc[post_idx]['post_id']),
                    'score': float(score),
                    'type': 'collaborative'
                }
                for post_idx, score in sorted(similar_user_posts.items(), key=lambda x: x[1], reverse=True)
                if self.user_item_matrix[user_id][post_idx] == 0  # Filter out posts user has already interacted with
            ][:n_recommendations]
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error in collaborative recommendations: {str(e)}")
            return self.get_popular_recommendations(n_recommendations)

    def get_popular_recommendations(self, n_recommendations: int = 5) -> List[Dict]:
        """
        Get popular posts as fallback recommendations
        """
        try:
            # Calculate post popularity scores
            popularity_scores = np.sum(self.user_item_matrix, axis=0)
            
            # Get top popular posts
            popular_posts = np.argsort(popularity_scores)[::-1][:n_recommendations]
            
            return [
                {
                    'post_id': int(self.post_features.iloc[i]['post_id']),
                    'score': float(popularity_scores[i]),
                    'type': 'popular'
                }
                for i in popular_posts
            ]
            
        except Exception as e:
            self.logger.error(f"Error getting popular recommendations: {str(e)}")
            return []

    def get_hybrid_recommendations(self, user_id: int, n_recommendations: int = 5) -> List[Dict]:
        """
        Get hybrid recommendations combining content-based and collaborative filtering
        """
        try:
            # Get recommendations from both approaches
            content_recs = self.get_content_based_recommendations(user_id, n_recommendations)
            collab_recs = self.get_collaborative_recommendations(user_id, n_recommendations)
            
            # Combine and normalize scores
            all_recs = defaultdict(lambda: {'score': 0, 'count': 0, 'types': set()})
            
            for rec in content_recs + collab_recs:
                post_id = rec['post_id']
                all_recs[post_id]['score'] += rec['score']
                all_recs[post_id]['count'] += 1
                all_recs[post_id]['types'].add(rec['type'])
            
            # Calculate final scores and sort
            final_recommendations = [
                {
                    'post_id': post_id,
                    'score': info['score'] / info['count'],
                    'types': list(info['types'])
                }
                for post_id, info in all_recs.items()
            ]
            
            final_recommendations.sort(key=lambda x: x['score'], reverse=True)
            
            return final_recommendations[:n_recommendations]
            
        except Exception as e:
            self.logger.error(f"Error in hybrid recommendations: {str(e)}")
            return self.get_popular_recommendations(n_recommendations)

    def add_new_interaction(self, user_id: int, post_id: int, interaction_type: str):
        """
        Process new user interaction and update models if needed
        """
        try:
            # Update Redis cache for real-time tracking
            interaction_key = f"interaction:{user_id}:{post_id}"
            self.redis_client.hincrby(interaction_key, interaction_type, 1)
            
            # Invalidate recommendation caches for this user
            self.redis_client.delete(f"content_recs:{user_id}")
            
            # Log the interaction for batch processing
            self.logger.info(f"New interaction logged: user={user_id}, post={post_id}, type={interaction_type}")
            
        except Exception as e:
            self.logger.error(f"Error processing new interaction: {str(e)}")