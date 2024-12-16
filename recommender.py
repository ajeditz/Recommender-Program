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
        # Initialize Redis for caching
        self.redis_client = redis.Redis(host=redis_host, port=redis_port)
        self.cache_ttl = 3600  # Cache TTL in seconds
        
        # Initialize storage for our models
        self.post_vectors = None
        self.user_item_matrix = None
        self.post_features = None
        self.similarity_matrix = None
        
        # Configure logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def process_post_content(self, posts_df: pd.DataFrame) -> np.ndarray:
        """
        Process post content using TF-IDF vectorization
        """
        # Combine relevant text fields
        posts_df['combined_text'] = posts_df['title']+" "+posts_df['description']+" "+posts_df['tags']+" "+ posts_df['location']
        # posts_df['combined_text']= posts_df['combined_text'].apply(lambda x: " ".join(x))
        
        
        # Create TF-IDF vectors
        vectorizer = TfidfVectorizer(
            max_features=100,
            stop_words='english',
            ngram_range=(1, 2)
        )
        # print(vectorizer)
        vector= vectorizer.fit_transform(posts_df['combined_text'])
        # print vectorizer.get_feature_names()
        return vector

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

    def get_content_based_recommendations(self, user_id: int, n_recommendations: int = 5) -> List[Dict]:
        """
        Get content-based recommendations based on user's interaction history
        """
        # Check cache first
        cache_key = f"content_recs:{user_id}"
        cached_recs = self.redis_client.get(cache_key)
        if cached_recs:
            return json.loads(cached_recs)

        try:
            # Get user's interaction history
            user_interactions = self.user_item_matrix[user_id]
            
            # Find posts user has interacted with
            interacted_posts = np.where(user_interactions > 0)[0]
            
            if len(interacted_posts) == 0:
                return self.get_popular_recommendations(n_recommendations)
            
            # Calculate average similarity with interacted posts
            sim_scores = np.mean([self.similarity_matrix[i] for i in interacted_posts], axis=0)
            
            # Get top similar posts
            similar_posts = np.argsort(sim_scores)[::-1]
            
            # Filter out already interacted posts
            recommendations = [
                {
                    'post_id': int(self.post_features.iloc[i]['post_id']),
                    'score': float(sim_scores[i]),
                    'type': 'content'
                }
                for i in similar_posts
                if i not in interacted_posts
            ][:n_recommendations]
            
            # Cache the results
            self.redis_client.setex(cache_key, self.cache_ttl, json.dumps(recommendations))
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error in content-based recommendations: {str(e)}")
            return self.get_popular_recommendations(n_recommendations)

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

            return "Interaction added successfully"
            
        except Exception as e:
            self.logger.error(f"Error processing new interaction: {str(e)}")