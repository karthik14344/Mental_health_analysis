import os
import re
import pandas as pd
import numpy as np
import folium
from folium import plugins
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
from utils.logging_config import setup_logging

logger = setup_logging()

class CrisisGeolocationMapper:
    def __init__(self):
        """Initialize the geolocation mapper with necessary components"""
        try:
            self.geolocator = Nominatim(user_agent="mental_health_crisis_mapper")
            
            # First, initialize country filters
            self._initialize_country_filters()
            
            # Then initialize country patterns
            self._initialize_country_patterns()
        except Exception as e:
            logger.error(f"Error initializing crisis mapper: {e}")
            raise

    def _initialize_country_filters(self):
        """Initialize comprehensive list of countries"""
        self.valid_countries = {
            # Continents
            'Afghanistan', 'Albania', 'Algeria', 'Andorra', 'Angola', 'Antigua and Barbuda', 
            'Argentina', 'Armenia', 'Australia', 'Austria', 'Azerbaijan', 'Bahamas', 'Bahrain', 
            'Bangladesh', 'Barbados', 'Belarus', 'Belgium', 'Belize', 'Benin', 'Bhutan', 
            'Bolivia', 'Bosnia and Herzegovina', 'Botswana', 'Brazil', 'Brunei', 'Bulgaria', 
            'Burkina Faso', 'Burundi', 'Cabo Verde', 'Cambodia', 'Cameroon', 'Canada', 
            'Central African Republic', 'Chad', 'Chile', 'China', 'Colombia', 'Comoros', 
            'Congo', 'Costa Rica', 'Croatia', 'Cuba', 'Cyprus', 'Czech Republic', 
            'Democratic Republic of the Congo', 'Denmark', 'Djibouti', 'Dominica', 
            'Dominican Republic', 'Ecuador', 'Egypt', 'El Salvador', 'Equatorial Guinea', 
            'Eritrea', 'Estonia', 'Eswatini', 'Ethiopia', 'Fiji', 'Finland', 'France', 
            'Gabon', 'Gambia', 'Georgia', 'Germany', 'Ghana', 'Greece', 'Grenada', 
            'Guatemala', 'Guinea', 'Guinea-Bissau', 'Guyana', 'Haiti', 'Honduras', 
            'Hungary', 'Iceland', 'India', 'Indonesia', 'Iran', 'Iraq', 'Ireland', 
            'Israel', 'Italy', 'Ivory Coast', 'Jamaica', 'Japan', 'Jordan', 'Kazakhstan', 
            'Kenya', 'Kiribati', 'Kosovo', 'Kuwait', 'Kyrgyzstan', 'Laos', 'Latvia', 
            'Lebanon', 'Lesotho', 'Liberia', 'Libya', 'Liechtenstein', 'Lithuania', 
            'Luxembourg', 'Madagascar', 'Malawi', 'Malaysia', 'Maldives', 'Mali', 
            'Malta', 'Marshall Islands', 'Mauritania', 'Mauritius', 'Mexico', 'Micronesia', 
            'Moldova', 'Monaco', 'Mongolia', 'Montenegro', 'Morocco', 'Mozambique', 
            'Myanmar', 'Namibia', 'Nauru', 'Nepal', 'Netherlands', 'New Zealand', 
            'Nicaragua', 'Niger', 'Nigeria', 'North Korea', 'North Macedonia', 'Norway', 
            'Oman', 'Pakistan', 'Palau', 'Palestine', 'Panama', 'Papua New Guinea', 
            'Paraguay', 'Peru', 'Philippines', 'Poland', 'Portugal', 'Qatar', 'Romania', 
            'Russia', 'Rwanda', 'Saint Kitts and Nevis', 'Saint Lucia', 
            'Saint Vincent and the Grenadines', 'Samoa', 'San Marino', 'Sao Tome and Principe', 
            'Saudi Arabia', 'Senegal', 'Serbia', 'Seychelles', 'Sierra Leone', 'Singapore', 
            'Slovakia', 'Slovenia', 'Solomon Islands', 'Somalia', 'South Africa', 
            'South Korea', 'South Sudan', 'Spain', 'Sri Lanka', 'Sudan', 'Suriname', 
            'Sweden', 'Switzerland', 'Syria', 'Taiwan', 'Tajikistan', 'Tanzania', 
            'Thailand', 'Timor-Leste', 'Togo', 'Tonga', 'Trinidad and Tobago', 'Tunisia', 
            'Turkey', 'Turkmenistan', 'Tuvalu', 'Uganda', 'Ukraine', 'United Arab Emirates', 
            'United Kingdom', 'United States', 'Uruguay', 'Uzbekistan', 'Vanuatu', 
            'Vatican City', 'Venezuela', 'Vietnam', 'Yemen', 'Zambia', 'Zimbabwe'
        }

        # Add common alternative names and abbreviations
        self.country_aliases = {
            # United States variations
            'USA': 'United States',
            'US': 'United States',
            'us': 'United States',
            'U.S.': 'United States',
            'U.S.A.': 'United States',
            'usa': 'United States',
            'united states': 'United States',
            'united states of america': 'United States',
            'america': 'United States',
            'united states america': 'United States',
            
            # United Kingdom variations
            'UK': 'United Kingdom',
            'uk': 'United Kingdom',
            'U.K.': 'United Kingdom',
            'u.k.': 'United Kingdom',
            'united kingdom': 'United Kingdom',
            'great britain': 'United Kingdom',
            'britain': 'United Kingdom',
            'england': 'United Kingdom',
            'british isles': 'United Kingdom',
            
            # United Arab Emirates variations
            'UAE': 'United Arab Emirates',
            'uae': 'United Arab Emirates',
            'u.a.e.': 'United Arab Emirates',
            'united arab emirates': 'United Arab Emirates',
            
            # Country name variations and common misspellings
            'germany': 'Germany',
            'germania': 'Germany',
            'deutsch': 'Germany',
            'federal republic of germany': 'Germany',
            
            'france': 'France',
            'french republic': 'France',
            'french': 'France',
            'république française': 'France',
            
            'canada': 'Canada',
            'canadian': 'Canada',
            
            'mexico': 'Mexico',
            'mexican': 'Mexico',
            'united mexican states': 'Mexico',
            'estados unidos mexicanos': 'Mexico',
            
            'brazil': 'Brazil',
            'brasil': 'Brazil',
            'federative republic of brazil': 'Brazil',
            
            'india': 'India',
            'bharat': 'India',
            'republic of india': 'India',
            
            'china': 'China',
            'prc': 'China',
            "people's republic of china": 'China',
            'chinese': 'China',
            
            'japan': 'Japan',
            'nippon': 'Japan',
            'nihon': 'Japan',
            
            'russia': 'Russia',
            'russian federation': 'Russia',
            'russian': 'Russia',
            
            'australia': 'Australia',
            'commonwealth of australia': 'Australia',
            'aussie': 'Australia',
            
            'spain': 'Spain',
            'reino de españa': 'Spain',
            'spanish': 'Spain',
            
            # Korea variations
            'Republic of Korea': 'South Korea',
            'south korea': 'South Korea',
            'korea': 'South Korea',
            'rok': 'South Korea',
            'korean': 'South Korea',
            
            'North Korea': 'North Korea',
            'dprk': 'North Korea',
            "democratic people's republic of korea": 'North Korea',
            'north korean': 'North Korea',
            
            # China variations
            'Republic of China': 'Taiwan',
            'taiwan': 'Taiwan',
            'formosa': 'Taiwan',
            'taiwanese': 'Taiwan',
            
            # More comprehensive list of global countries
            'argentina': 'Argentina',
            'argentine republic': 'Argentina',
            
            'ukraine': 'Ukraine',
            'ukrainian': 'Ukraine',
            
            'turkey': 'Turkey',
            'turkish republic': 'Turkey',
            
            'switzerland': 'Switzerland',
            'swiss': 'Switzerland',
            'swiss confederation': 'Switzerland',
            
            'italy': 'Italy',
            'italian republic': 'Italy',
            
            'egypt': 'Egypt',
            'arab republic of egypt': 'Egypt',
            
            'greece': 'Greece',
            'hellenic republic': 'Greece',
            
            'south africa': 'South Africa',
            'republic of south africa': 'South Africa',
            
            'poland': 'Poland',
            'republic of poland': 'Poland',
            
            'saudi arabia': 'Saudi Arabia',
            'kingdom of saudi arabia': 'Saudi Arabia',
            
            'iran': 'Iran',
            'islamic republic of iran': 'Iran',
            
            'iraq': 'Iraq',
            'republic of iraq': 'Iraq',
            
            'pakistan': 'Pakistan',
            'islamic republic of pakistan': 'Pakistan',
            
            'nigeria': 'Nigeria',
            'federal republic of nigeria': 'Nigeria',
            
            'peru': 'Peru',
            'republic of peru': 'Peru',
            
            # Demonyms and alternative references
            'mexican': 'Mexico',
            'american': 'United States',
            'british': 'United Kingdom',
            'canadian': 'Canada',
            'chinese': 'China',
            'japanese': 'Japan',
            'indian': 'India',
            'russian': 'Russia',
            'french': 'France',
            'german': 'Germany',
            'spanish': 'Spain',
            'italian': 'Italy'
        }
        
        self.invalid_locations = {
            'my life', 'the last', 'life', 'me', 'my world',
            'everywhere', 'anywhere', 'nowhere', 'somewhere'
        }

    def _initialize_country_patterns(self):
        """Initialize comprehensive regex patterns for country extraction"""
        # Escape special characters in country names for regex
        escaped_countries = [re.escape(country) for country in list(self.valid_countries) + list(self.country_aliases.keys())]
        
        # Create regex patterns with various contextual mentions
        context_words = r'(in|from|at|of|near|around|located\s+in)'
        
        self.country_patterns = [
            # Pattern with context words
            rf'\b{context_words}\s+({"|".join(escaped_countries)})\b',
            
            # Direct country mentions
            rf'\b({"|".join(escaped_countries)})\b',
            
            # Broader context patterns
            r'\b(living\s+in|traveling\s+to|visiting)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b'
        ]

    def extract_countries(self, text):
        """
        Extract potential countries from text using comprehensive regex patterns
        """
        countries = set()
        
        for pattern in self.country_patterns:
            # Find matches
            matches = re.findall(pattern, text, re.IGNORECASE)
            
            for match in matches:
                # Handle different match structures
                if isinstance(match, tuple):
                    # If match is a tuple, take the last element (country name)
                    country = match[-1]
                else:
                    country = match
                
                # Normalize country name
                country = country.strip()
                
                # Check against valid countries and aliases
                normalized_country = self.country_aliases.get(country, country)
                
                if self.is_valid_country(normalized_country):
                    countries.add(normalized_country)
        
        return list(countries)

    def is_valid_country(self, country):
        """
        Validate if extracted country is legitimate with comprehensive case-insensitive checks
        """
        # Convert to string and lowercase for comparison
        country_str = str(country).lower().strip()
        
        # Check against invalid locations (case-insensitive)
        if country_str in (loc.lower() for loc in self.invalid_locations):
            return False
        
        # Check against country aliases (case-insensitive)
        if country_str in (alias.lower() for alias in self.country_aliases):
            return True
        
        # Normalize aliases to handle case variations
        normalized_aliases = {alias.lower(): original for alias, original in self.country_aliases.items()}
        if country_str in normalized_aliases:
            return True
        
        # Check against valid countries (case-insensitive)
        return any(country_str == valid_country.lower() for valid_country in self.valid_countries)
      
    def extract_countries(self, text):
        """
        Extract potential countries from text using regex patterns with validation
        """
        countries = []
        for pattern in self.country_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                # Handle tuple or single string match
                country = match[1] if isinstance(match, tuple) else match
                if self.is_valid_country(country):
                    countries.append(country)
        return list(set(countries))  # Remove duplicates

    def geocode_country(self, country):
        """
        Geocode a country to get its centroid coordinates
        """
        try:
            if not self.is_valid_country(country):
                return None

            # Attempt geocoding
            location_data = self.geolocator.geocode(
                country, 
                timeout=10,
                exactly_one=True
            )

            if location_data:
                return {
                    'country': country,
                    'latitude': location_data.latitude,
                    'longitude': location_data.longitude,
                    'confidence': location_data.raw.get('importance', 0)
                }

        except (GeocoderTimedOut, GeocoderServiceError) as e:
            logger.warning(f"Geocoding error for {country}: {e}")
        except Exception as e:
            logger.error(f"Unexpected error geocoding {country}: {e}")

        return None

    def process_geolocation(self, dataframe):
        """
        Process dataframe with country-level geocoding
        """
        valid_locations = []
        
        for _, row in dataframe.iterrows():
            extracted_countries = self.extract_countries(row['content'])
            
            for country in extracted_countries:
                geo_result = self.geocode_country(country)
                if geo_result:
                    post_data = row.to_dict()
                    post_data.update(geo_result)
                    valid_locations.append(post_data)
        
        if valid_locations:
            return pd.DataFrame(valid_locations)
        else:
            logger.warning("No valid countries found in the dataset")
            return pd.DataFrame()

    def create_crisis_heatmap(self, geolocated_df, output_filename='crisis_country_heatmap.html'):
        """
        Generate an interactive heatmap of crisis discussions by country with all countries
        """
        # Create base map centered on the world
        crisis_map = folium.Map(location=[0, 0], zoom_start=2)
        
        # Count discussions per country
        country_counts = geolocated_df.groupby('country').size()
        
        # Normalize heat intensity based on country frequencies
        max_count = country_counts.max()
        heat_data = []
        
        # Add markers and heatmap data for all countries
        for country, count in country_counts.items():
            country_data = geolocated_df[geolocated_df['country'] == country].iloc[0]
            
            # Calculate heat intensity (proportional to the number of discussions)
            heat_intensity = (count / max_count) * 10  # Scale from 0 to 10
            
            # Add to heatmap data
            heat_data.append([
                country_data['latitude'], 
                country_data['longitude'], 
                heat_intensity
            ])
            
            # Add marker with tooltip showing count
            folium.Marker(
                [country_data['latitude'], country_data['longitude']],
                popup=f"{country}: {count} crisis discussions",
                tooltip=f"{country} - {count} posts"
            ).add_to(crisis_map)
        
        # Add heatmap layer with variable intensity
        folium.plugins.HeatMap(
            heat_data, 
            name='Crisis Discussion Intensity', 
            radius=25,
            blur=15,
            max_zoom=1
        ).add_to(crisis_map)
        
        # Ensure output directory exists
        os.makedirs('output', exist_ok=True)
        output_path = os.path.join('output', output_filename)
        
        # Save heatmap
        crisis_map.save(output_path)
        print(f"Crisis country heatmap saved to {output_path}")
        
        # Return country counts for reporting
        return country_counts
