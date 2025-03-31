import os
import pandas as pd
import numpy as np
from textblob import TextBlob
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import gensim.downloader as api
import logging
from utils.logging_config import setup_logging

logger = setup_logging()

class EnhancedMentalHealthRiskClassifier:
    def __init__(self):
        """Initialize the risk classifier with necessary models and terms"""
        try:
            # Download NLTK data if not already present
            nltk.download('punkt', quiet=True)
            
            self._initialize_risk_terms()
            self._initialize_models()
            
        except Exception as e:
            logger.error(f"Error initializing risk classifier: {e}")
            raise

    def _initialize_risk_terms(self):
        """Initialize risk terms for classification"""
        self.high_risk_terms = [
            "suicide", "kill myself", "want to die", "end it all",
            "no hope", "can't go on", "death", "suicidal thoughts",
            "self-harm", "cutting", "overdose", "i want to die"
        ]
        
        self.moderate_risk_terms = [
            "struggling", "help", "can't cope", "overwhelmed",
            "depression", "anxiety", "mental breakdown",
            "feeling lost", "ptsd", "burnout", "addiction",
            "substance abuse", "emotional pain"
        ]

    def _initialize_models(self):
        """Initialize NLP models"""
        try:
            self.tfidf_vectorizer = TfidfVectorizer(stop_words='english')
            self.word2vec_model = api.load('word2vec-google-news-300')
        except Exception as e:
            logger.error(f"Error loading Word2Vec model: {e}")
            self.word2vec_model = None

    def classify_sentiment(self, text):
        """Classify sentiment using TextBlob with enhanced granularity"""
        try:
            if not isinstance(text, str) or not text.strip():
                return 'Neutral'
                
            blob = TextBlob(str(text))
            polarity = blob.sentiment.polarity
            
            if polarity > 0.3: return 'Very Positive'
            elif polarity > 0.1: return 'Positive'
            elif polarity < -0.3: return 'Very Negative'
            elif polarity < -0.1: return 'Negative'
            return 'Neutral'
            
        except Exception as e:
            logger.error(f"Error in sentiment classification: {e}")
            return 'Neutral'

    def tfidf_risk_detection(self, text, risk_terms):
        """
        Use TF-IDF to detect risk-related terms
        """
        # Combine text with risk terms
        corpus = [text] + risk_terms
        
        # Compute TF-IDF
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(corpus)
        
        # Compute cosine similarity between text and risk terms
        similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])[0]
        
        # If any similarity is high, consider it a match
        return max(similarities) > 0.3

    def word2vec_risk_detection(self, text, risk_terms):
        """
        Use Word2Vec for semantic similarity detection
        """
        if self.word2vec_model is None:
            return False
        
        # Tokenize text
        text_words = text.lower().split()
        
        # Compute average vector for text
        try:
            text_vector = np.mean([self.word2vec_model[word] for word in text_words 
                                   if word in self.word2vec_model], axis=0)
        except:
            return False
        
        # Check semantic similarity with risk terms
        for term in risk_terms:
            term_words = term.lower().split()
            try:
                term_vector = np.mean([self.word2vec_model[word] for word in term_words 
                                       if word in self.word2vec_model], axis=0)
                
                # Compute cosine similarity
                similarity = np.dot(text_vector, term_vector) / (
                    np.linalg.norm(text_vector) * np.linalg.norm(term_vector)
                )
                
                if similarity > 0.7:  # High semantic similarity threshold
                    return True
            except:
                continue
        
        return False

    def classify_risk_level(self, text):
        """
        Enhanced risk classification using multiple techniques
        """
        text_lower = str(text).lower()
        
        # Check for exact keyword matches
        for term in self.high_risk_terms:
            if term in text_lower:
                return 'High Risk'
        
        # TF-IDF Risk Detection
        if self.tfidf_risk_detection(text, self.high_risk_terms):
            return 'High Risk'
        
        # Word2Vec Semantic Similarity
        if self.word2vec_risk_detection(text, self.high_risk_terms):
            return 'High Risk'
        
        # Similar process for moderate risk
        for term in self.moderate_risk_terms:
            if term in text_lower:
                return 'Moderate Concern'
        
        if self.tfidf_risk_detection(text, self.moderate_risk_terms):
            return 'Moderate Concern'
        
        if self.word2vec_risk_detection(text, self.moderate_risk_terms):
            return 'Moderate Concern'
        
        return 'Low Concern'

    def process_posts(self, posts_dataframe):
        """
        Process posts and add sentiment and risk level columns
        """
        # Add sentiment column
        posts_dataframe['Sentiment'] = posts_dataframe['content'].apply(self.classify_sentiment)
        
        # Add risk level column
        posts_dataframe['Risk Level'] = posts_dataframe['content'].apply(self.classify_risk_level)
        
        return posts_dataframe

    def generate_risk_distribution(self, processed_df):
        """
        Generate distribution of posts by sentiment and risk level
        """
        # Create cross-tabulation
        risk_distribution = pd.crosstab(
            processed_df['Sentiment'], 
            processed_df['Risk Level']
        )
        
        return risk_distribution

    def save_results(self, processed_df, distribution, base_filename='mental_health_analysis'):
        """
        Save processed results to CSV and generate visualization
        """
        # Create output directory
        os.makedirs('output', exist_ok=True)
        
        # Save processed dataframe
        processed_filename = os.path.join('output', f'{base_filename}_processed.csv')
        processed_df.to_csv(processed_filename, index=False)
        print(f"Processed data saved to {processed_filename}")
        
        # Save distribution
        dist_filename = os.path.join('output', f'{base_filename}_distribution.csv')
        distribution.to_csv(dist_filename)
        print(f"Risk distribution saved to {dist_filename}")
        
        # Create heatmap visualization
        plt.figure(figsize=(10, 6))
        sns.heatmap(distribution, annot=True, cmap='YlGnBu', fmt='g')
        plt.title('Mental Health Posts: Sentiment vs Risk Level')
        plt.tight_layout()
        
        # Save heatmap
        heatmap_filename = os.path.join('output', f'{base_filename}_heatmap.png')
        plt.savefig(heatmap_filename)
        print(f"Heatmap visualization saved to {heatmap_filename}")
        
        return distribution
      
      


