import os
from dotenv import load_dotenv
from risk_classifier import EnhancedMentalHealthRiskClassifier
from reddit_mental_health_extractor import RedditMentalHealthDataExtractor
from crisis_geolocation_mapper import CrisisGeolocationMapper
from utils.logging_config import setup_logging
import pandas as pd

logger = setup_logging()

def main():
    """Main execution function for mental health analysis and risk classification"""
    try:
        # Load environment variables and validate
        load_dotenv()
        required_vars = ['CLIENT_ID', 'CLIENT_SECRET', 'USER_AGENT']
        if not all(os.getenv(var) for var in required_vars):
            raise ValueError("Missing required environment variables")
        
        # Initialize components
        logger.info("Initializing system components...")
        extractor = RedditMentalHealthDataExtractor(
            os.getenv('CLIENT_ID'),
            os.getenv('CLIENT_SECRET'),
            os.getenv('USER_AGENT')
        )
        classifier = EnhancedMentalHealthRiskClassifier()
        geolocation_mapper = CrisisGeolocationMapper()
        
        # Define target subreddits
        target_subreddits = [
            'mentalhealth', 'depression', 'anxiety',
            'addiction', 'SuicideWatch', 'therapy',
            'MentalHealthSupport', 'COVID19_support'
        ]
        
        # Extract and process data
        logger.info("Extracting posts from target subreddits...")
        posts = extractor.extract_posts(target_subreddits)
        posts_df = pd.DataFrame(posts)
        
        # Risk Classification
        logger.info("Performing risk classification...")
        processed_df = classifier.process_posts(posts_df)
        risk_distribution = classifier.generate_risk_distribution(processed_df)
        
        # Geolocation Analysis
        logger.info("Processing geolocation data...")
        geolocated_df = geolocation_mapper.process_geolocation(processed_df)
        
        # Save all results
        output_dir = 'output'
        os.makedirs(output_dir, exist_ok=True)
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        
        # 1. Save raw data
        raw_data_path = os.path.join(output_dir, f'raw_posts_{timestamp}.csv')
        posts_df.to_csv(raw_data_path, index=False)
        logger.info(f"Raw data saved to {raw_data_path}")
        
        # 2. Save processed data with risk classification
        processed_path = os.path.join(output_dir, f'processed_posts_{timestamp}.csv')
        processed_df.to_csv(processed_path, index=False)
        logger.info(f"Processed data saved to {processed_path}")
        
        # 3. Save risk distribution
        classifier.save_results(
            processed_df,
            risk_distribution,
            base_filename=f'risk_analysis_{timestamp}'
        )
        
        # 4. Save geolocation data
        geolocated_path = os.path.join(output_dir, f'geolocated_data_{timestamp}.csv')
        geolocated_df.to_csv(geolocated_path, index=False)
        logger.info(f"Geolocated data saved to {geolocated_path}")
        
        # Generate and save visualizations
        logger.info("Generating visualizations...")
        
        # 5. Create and save crisis heatmap
        top_locations = geolocation_mapper.create_crisis_heatmap(geolocated_df)
        
        # Generate summary report
        report_path = os.path.join(output_dir, f'analysis_report_{timestamp}.txt')
        with open(report_path, 'w') as f:
            f.write("Mental Health Analysis Report\n")
            f.write("===========================\n\n")
            f.write(f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Posts Analyzed: {len(processed_df)}\n\n")
            
            f.write("Risk Level Distribution:\n")
            risk_counts = processed_df['Risk Level'].value_counts()
            for risk_level, count in risk_counts.items():
                f.write(f"{risk_level}: {count} posts ({count/len(processed_df)*100:.1f}%)\n")
            
            f.write("\nTop 5 Locations with Highest Crisis Discussions:\n")
            f.write(str(top_locations))
            
            f.write("\nRisk-Sentiment Distribution:\n")
            f.write(str(risk_distribution))
        
        logger.info("Analysis completed successfully")
        logger.info(f"Full analysis report saved to {report_path}")
        logger.info("\nTop 5 Locations with Highest Crisis Discussions:")
        logger.info(top_locations)
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()
