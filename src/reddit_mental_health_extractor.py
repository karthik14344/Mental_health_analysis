import os
from dotenv import load_dotenv
import praw
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
import json
from datetime import datetime
from utils.logging_config import setup_logging

logger = setup_logging()

class RedditMentalHealthDataExtractor:
    def __init__(self, client_id, client_secret, user_agent):
        """Initialize the Reddit API client and required resources"""
        try:
            self.reddit = praw.Reddit(
                client_id=client_id,
                client_secret=client_secret,
                user_agent=user_agent
            )
            
            # Download NLTK resources if needed
            nltk.download('stopwords', quiet=True)
            
            self._initialize_keywords()
            self.stop_words = set(stopwords.words('english'))
            
        except Exception as e:
            logger.error(f"Error initializing Reddit extractor: {e}")
            raise

    def _initialize_keywords(self):
        """Initialize health-related keywords"""
        self.keywords = [
            'depressed', 'depression', 'anxiety', 'overwhelmed',
            'mental health', 'struggling', 'addiction', 'substance abuse',
            'suicidal', 'suicide', 'help', 'therapy', 'counseling',
            'mental breakdown', 'ptsd', 'burnout', 'stress',
            'emotional pain', 'mental struggle', 'coping'
        ]
        
        # Stopwords to remove
        self.stop_words = set(stopwords.words('english'))
    
    def clean_text(self, text):
       
        # Check if text is None or empty
        if not text or not isinstance(text, str):
            return ''
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove stopwords
        words = text.split()
        words = [word for word in words if word not in self.stop_words]
        
        return ' '.join(words)
    
    def extract_posts(self, subreddits, limit=200, min_upvotes=5):
      
        # Returns:list: List of dictionaries containing post information
      
        extracted_posts = []
        
        for subreddit_name in subreddits:
            try:
                subreddit = self.reddit.subreddit(subreddit_name)
                
                # Combine different post sorting methods with updated syntax
                post_generators = [
                    subreddit.hot(limit=limit),
                    subreddit.new(limit=limit),
                    subreddit.top(time_filter='week', limit=limit)  # Updated syntax
                ]
                
                processed_post_ids = set()
                
                for generator in post_generators:
                    for post in generator:
                        # Skip duplicates
                        if post.id in processed_post_ids:
                            continue
                        
                        # Check upvotes and keyword criteria
                        if (post.ups >= min_upvotes and 
                            (any(keyword.lower() in post.title.lower() or 
                                 (post.selftext and keyword.lower() in post.selftext.lower()) 
                                 for keyword in self.keywords))):
                            
                            post_data = {
                                'post_id': post.id,
                                'timestamp': datetime.fromtimestamp(post.created_utc).isoformat(),
                                'title': post.title,
                                'content': post.selftext or '',
                                'subreddit': subreddit_name,
                                'cleaned_content': self.clean_text(post.selftext),
                                'upvotes': post.ups,
                                'num_comments': post.num_comments,
                                'url': post.url
                            }
                            
                            extracted_posts.append(post_data)
                            processed_post_ids.add(post.id)
            
            except Exception as e:
                print(f"Error extracting posts from {subreddit_name}: {e}")
        
        return extracted_posts
    
    def save_data(self, posts, base_filename='reddit_mental_health_posts'):
        
        # Create output directory if it doesn't exist
        os.makedirs('output', exist_ok=True)
        
        # Generate timestamp for unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # CSV Output
        csv_filename = os.path.join('output', f'{base_filename}_{timestamp}.csv')
        df = pd.DataFrame(posts)
        df.to_csv(csv_filename, index=False)
        print(f"Saved {len(posts)} posts to {csv_filename}")
        
        # JSON Output
        json_filename = os.path.join('output', f'{base_filename}_{timestamp}.json')
        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump(posts, f, ensure_ascii=False, indent=4)
        print(f"Saved {len(posts)} posts to {json_filename}")

def main():
    
    client_id = os.getenv('CLIENT_ID')
    client_secret = os.getenv('CLIENT_SECRET')
    user_agent = os.getenv('USER_AGENT')

    
    # Subreddits to search
    target_subreddits = [
        'mentalhealth', 'depression', 'anxiety', 
        'addiction', 'SuicideWatch', 'therapy',
        'MentalHealthSupport', 'COVID19_support'
    ]
    
    # Initialize extractor
    extractor = RedditMentalHealthDataExtractor(client_id, client_secret, user_agent)
    
    # Extract posts
    posts = extractor.extract_posts(target_subreddits, limit=250, min_upvotes=5)
    
    # Save data
    extractor.save_data(posts)

if __name__ == '__main__':
    main()
