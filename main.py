"""
Streaming Platform Recommendation System
Built with FastAPI, Scikit-learn, and Gemini LLM
"""

import pandas as pd
import numpy as np
from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import HTMLResponse
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import warnings
import os
from typing import List, Dict, Optional
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

warnings.filterwarnings('ignore')

# Initialize FastAPI app
app = FastAPI(title="Streaming Recommendation System")

# Global variables to store data and models
users_df = None
items_df = None
events_df = None
user_item_matrix = None
item_similarity_matrix = None
popular_items = None

# ==================== DATA LOADING & CLEANSING ====================

def load_and_cleanse_data():
    """Load CSV files and perform data cleansing"""
    global users_df, items_df, events_df

    print("Loading data...")

    # Load datasets
    users_df = pd.read_csv('users.csv')
    items_df = pd.read_csv('items.csv')
    events_df = pd.read_csv('events.csv')

    print(f"Initial data shape - Users: {users_df.shape}, Items: {items_df.shape}, Events: {events_df.shape}")

    # Data Cleansing - Check for null values
    print("\n=== Data Cleansing Report ===")

    # Users cleansing
    print(f"\nUsers null values:\n{users_df.isnull().sum()}")
    users_df = users_df.dropna(subset=['user_id'])  # Drop if user_id is null
    users_df['age'] = users_df['age'].fillna(users_df['age'].median())  # Fill age with median
    users_df['gender'] = users_df['gender'].fillna('Unknown')
    users_df['region'] = users_df['region'].fillna('Unknown')

    # Items cleansing
    print(f"\nItems null values:\n{items_df.isnull().sum()}")
    items_df = items_df.dropna(subset=['item_id', 'title'])  # Drop if essential fields are null
    items_df['content_type'] = items_df['content_type'].fillna('unknown')
    items_df['genre'] = items_df['genre'].fillna('unknown')

    # Events cleansing
    print(f"\nEvents null values:\n{events_df.isnull().sum()}")
    events_df = events_df.dropna(subset=['user_id', 'item_id'])  # Drop if essential fields are null
    events_df['watch_seconds'] = events_df['watch_seconds'].fillna(0)
    events_df['event_type'] = events_df['event_type'].fillna('play')

    # Remove events with invalid user_id or item_id
    valid_users = set(users_df['user_id'])
    valid_items = set(items_df['item_id'])
    initial_events = len(events_df)
    events_df = events_df[events_df['user_id'].isin(valid_users) & events_df['item_id'].isin(valid_items)]
    print(f"\nRemoved {initial_events - len(events_df)} invalid events")

    print(f"\nCleaned data shape - Users: {users_df.shape}, Items: {items_df.shape}, Events: {events_df.shape}")
    print("=== Data Cleansing Complete ===\n")

    return users_df, items_df, events_df


# ==================== RECOMMENDATION MODELS ====================

def calculate_popular_items():
    """Calculate global popularity scores"""
    global popular_items

    # Weight different event types
    event_weights = {
        'play': 1.0,
        'complete': 3.0,
        'like': 2.5,
        'save': 2.0,
        'pause': 0.5,
        'skip': 0.1
    }

    # Calculate weighted score
    events_df['weighted_score'] = events_df.apply(
        lambda row: row['watch_seconds'] * event_weights.get(row['event_type'], 1.0),
        axis=1
    )

    # Aggregate by item
    popularity = events_df.groupby('item_id').agg({
        'weighted_score': 'sum',
        'user_id': 'nunique',  # unique users
        'event_type': 'count'  # total events
    }).reset_index()

    popularity.columns = ['item_id', 'total_score', 'unique_users', 'total_events']

    # Normalize and create final popularity score
    popularity['popularity_score'] = (
        0.5 * (popularity['total_score'] / popularity['total_score'].max()) +
        0.3 * (popularity['unique_users'] / popularity['unique_users'].max()) +
        0.2 * (popularity['total_events'] / popularity['total_events'].max())
    )

    # Merge with item details
    popular_items = popularity.merge(items_df, on='item_id', how='left')
    popular_items = popular_items.sort_values('popularity_score', ascending=False)

    return popular_items


def build_user_item_matrix():
    """Build user-item interaction matrix for collaborative filtering"""
    global user_item_matrix

    # Create interaction score based on watch_seconds and event_type
    event_weights = {
        'play': 1.0,
        'complete': 3.0,
        'like': 2.5,
        'save': 2.0,
        'pause': 0.5,
        'skip': 0.1
    }

    events_df['interaction_score'] = events_df.apply(
        lambda row: row['watch_seconds'] * event_weights.get(row['event_type'], 1.0),
        axis=1
    )

    # Aggregate multiple interactions
    interaction_df = events_df.groupby(['user_id', 'item_id'])['interaction_score'].sum().reset_index()

    # Create pivot table (user-item matrix)
    user_item_matrix = interaction_df.pivot_table(
        index='user_id',
        columns='item_id',
        values='interaction_score',
        fill_value=0
    )

    return user_item_matrix


def build_item_similarity_matrix():
    """Build item-item similarity matrix using cosine similarity"""
    global item_similarity_matrix

    # Normalize the user-item matrix
    normalized_matrix = normalize(user_item_matrix.values, axis=0)

    # Calculate cosine similarity between items
    item_similarity_matrix = cosine_similarity(normalized_matrix.T)

    # Convert to DataFrame
    item_similarity_matrix = pd.DataFrame(
        item_similarity_matrix,
        index=user_item_matrix.columns,
        columns=user_item_matrix.columns
    )

    return item_similarity_matrix


def recommend_popular(k: int = 10) -> List[Dict]:
    """
    Return top-k items by global popularity
    """
    top_items = popular_items.head(k)

    recommendations = []
    for _, row in top_items.iterrows():
        recommendations.append({
            'item_id': row['item_id'],
            'title': row['title'],
            'content_type': row['content_type'],
            'genre': row['genre'],
            'score': round(row['popularity_score'], 4),
            'reason': 'Globally popular content'
        })

    return recommendations


def recommend_for_user(user_id: str, k: int = 10) -> tuple:
    """
    Returns personalized recommendations for a user
    Returns: (recommendations_list, fallback_used)
    """
    # Check if user exists and has history
    if user_id not in user_item_matrix.index:
        return recommend_popular(k), True

    user_interactions = user_item_matrix.loc[user_id]

    # Check if user has any interactions
    if user_interactions.sum() == 0:
        return recommend_popular(k), True

    # Get items user has already watched significantly (threshold: top 30% of user's watch time)
    watched_items = user_interactions[user_interactions > user_interactions.quantile(0.7)].index.tolist()

    # Calculate recommendation scores using item-based collaborative filtering
    recommendation_scores = {}

    for item in watched_items:
        if item in item_similarity_matrix.index:
            # Get similar items
            similar_items = item_similarity_matrix[item].sort_values(ascending=False)

            # Weight by user's interaction with the source item
            weight = user_interactions[item]

            for similar_item, similarity in similar_items.items():
                if similar_item not in watched_items:  # Don't recommend already watched
                    if similar_item not in recommendation_scores:
                        recommendation_scores[similar_item] = 0
                    recommendation_scores[similar_item] += similarity * weight

    # Sort by score
    sorted_recommendations = sorted(recommendation_scores.items(), key=lambda x: x[1], reverse=True)

    # If not enough recommendations, supplement with popular items
    if len(sorted_recommendations) < k:
        popular_recs = recommend_popular(k)
        remaining = k - len(sorted_recommendations)
        for rec in popular_recs:
            if rec['item_id'] not in [r[0] for r in sorted_recommendations] and rec['item_id'] not in watched_items:
                sorted_recommendations.append((rec['item_id'], rec['score']))
                remaining -= 1
                if remaining == 0:
                    break

    # Get top-k
    top_recommendations = sorted_recommendations[:k]

    # Format output
    recommendations = []
    for item_id, score in top_recommendations:
        item_info = items_df[items_df['item_id'] == item_id].iloc[0]

        # Find what similar item this was based on
        reason = "Based on your viewing history"
        for watched_item in watched_items[:3]:  # Check top 3 watched items
            if item_id in item_similarity_matrix.index and watched_item in item_similarity_matrix.columns:
                similarity = item_similarity_matrix.loc[item_id, watched_item]
                if similarity > 0.1:
                    watched_title = items_df[items_df['item_id'] == watched_item].iloc[0]['title']
                    reason = f"Similar to '{watched_title}'"
                    break

        recommendations.append({
            'item_id': item_id,
            'title': item_info['title'],
            'content_type': item_info['content_type'],
            'genre': item_info['genre'],
            'score': round(float(score), 4),
            'reason': reason
        })

    return recommendations, False


# ==================== GEMINI LLM INTEGRATION ====================

def init_gemini():
    """Initialize Gemini API"""
    try:
        # Try to get API key from environment variable
        api_key = os.getenv('GOOGLE_API_KEY')
        if api_key:
            genai.configure(api_key=api_key)
            print("✓ Gemini API initialized successfully")
            return True
        else:
            print("Warning: GOOGLE_API_KEY not found in .env file. LLM features will be disabled.")
            return False
    except Exception as e:
        print(f"Warning: Failed to initialize Gemini: {e}")
        return False


def enhance_recommendations_with_llm(user_id: str, recommendations: List[Dict]) -> str:
    """Use Gemini to generate personalized explanation"""
    try:
        # Get user info
        user_info = users_df[users_df['user_id'] == user_id].iloc[0] if user_id in users_df['user_id'].values else None

        # Get user's watch history
        user_history = events_df[events_df['user_id'] == user_id].merge(items_df, on='item_id')
        top_watched = user_history.nlargest(5, 'watch_seconds')[['title', 'genre', 'content_type']].to_dict('records')

        # Create prompt
        prompt = f"""
        You are a friendly streaming platform assistant. Create a brief, personalized message (2-3 sentences)
        explaining why these recommendations are great for the user.

        User Profile:
        - Age: {user_info['age'] if user_info is not None else 'Unknown'}
        - Gender: {user_info['gender'] if user_info is not None else 'Unknown'}
        - Region: {user_info['region'] if user_info is not None else 'Unknown'}

        User's Top Watched Content:
        {top_watched[:3]}

        Recommended Items:
        {[{'title': r['title'], 'genre': r['genre'], 'type': r['content_type']} for r in recommendations[:5]]}

        Write a warm, personalized message explaining these recommendations.
        """

        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(prompt)

        return response.text
    except Exception as e:
        return f"Based on your viewing preferences, we've curated these personalized recommendations just for you!"


# ==================== FASTAPI ENDPOINTS ====================

@app.on_event("startup")
async def startup_event():
    """Initialize the recommendation system on startup"""
    print("Starting Recommendation System...")

    # Load and cleanse data
    load_and_cleanse_data()

    # Build models
    print("Building recommendation models...")
    calculate_popular_items()
    build_user_item_matrix()
    build_item_similarity_matrix()

    # Initialize Gemini
    init_gemini()

    print("Recommendation System Ready!")


@app.get("/", response_class=HTMLResponse)
async def home():
    """Serve the frontend HTML"""
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Streaming Recommendation System</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
        <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
        <style>
            body {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            }
            .main-container {
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
            }
            .header {
                text-align: center;
                color: white;
                padding: 40px 0;
            }
            .header h1 {
                font-size: 3rem;
                font-weight: bold;
                margin-bottom: 10px;
                text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
            }
            .header p {
                font-size: 1.2rem;
                opacity: 0.9;
            }
            .search-card {
                background: white;
                border-radius: 20px;
                padding: 30px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.3);
                margin-bottom: 30px;
            }
            .btn-primary {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                border: none;
                padding: 12px 30px;
                font-size: 1.1rem;
                border-radius: 10px;
                transition: transform 0.2s;
            }
            .btn-primary:hover {
                transform: translateY(-2px);
                box-shadow: 0 5px 15px rgba(0,0,0,0.3);
            }
            .btn-secondary {
                background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
                border: none;
                padding: 12px 30px;
                font-size: 1.1rem;
                border-radius: 10px;
                transition: transform 0.2s;
            }
            .btn-secondary:hover {
                transform: translateY(-2px);
                box-shadow: 0 5px 15px rgba(0,0,0,0.3);
            }
            .results-card {
                background: white;
                border-radius: 20px;
                padding: 30px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.3);
                display: none;
            }
            .recommendation-item {
                background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
                border-radius: 15px;
                padding: 20px;
                margin-bottom: 15px;
                transition: transform 0.2s;
                border-left: 5px solid #667eea;
            }
            .recommendation-item:hover {
                transform: translateX(5px);
                box-shadow: 0 5px 15px rgba(0,0,0,0.2);
            }
            .item-title {
                font-size: 1.3rem;
                font-weight: bold;
                color: #2d3748;
                margin-bottom: 10px;
            }
            .item-meta {
                display: flex;
                gap: 15px;
                margin-bottom: 10px;
            }
            .badge-custom {
                padding: 5px 15px;
                border-radius: 20px;
                font-size: 0.9rem;
            }
            .item-reason {
                color: #718096;
                font-style: italic;
                margin-top: 10px;
            }
            .llm-message {
                background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
                border-radius: 15px;
                padding: 20px;
                margin-bottom: 20px;
                border-left: 5px solid #f5576c;
            }
            .loading {
                text-align: center;
                padding: 40px;
                display: none;
            }
            .spinner-border {
                width: 3rem;
                height: 3rem;
            }
            .form-control {
                border-radius: 10px;
                padding: 12px;
                font-size: 1.1rem;
            }
            .input-group {
                margin-bottom: 20px;
            }
        </style>
    </head>
    <body>
        <div class="main-container">
            <div class="header">
                <h1><i class="fas fa-tv"></i> Streaming Recommendations</h1>
                <p>Discover your next favorite show powered by AI</p>
            </div>

            <div class="search-card">
                <h3 class="mb-4"><i class="fas fa-search"></i> Find Recommendations</h3>

                <div class="input-group mb-3">
                    <span class="input-group-text"><i class="fas fa-user"></i></span>
                    <input type="text" class="form-control" id="userId" placeholder="Enter User ID (e.g., u1)">
                </div>

                <div class="input-group mb-3">
                    <span class="input-group-text"><i class="fas fa-list-ol"></i></span>
                    <input type="number" class="form-control" id="numRecs" placeholder="Number of recommendations" value="10" min="1" max="50">
                </div>

                <div class="d-grid gap-2 d-md-flex justify-content-md-center">
                    <button class="btn btn-primary btn-lg" onclick="getUserRecommendations()">
                        <i class="fas fa-star"></i> Get My Recommendations
                    </button>
                    <button class="btn btn-secondary btn-lg" onclick="getPopularRecommendations()">
                        <i class="fas fa-fire"></i> Show Popular
                    </button>
                </div>
            </div>

            <div class="loading" id="loading">
                <div class="spinner-border text-light" role="status">
                    <span class="visually-hidden">Loading...</span>
                </div>
                <p class="text-light mt-3">Finding perfect recommendations for you...</p>
            </div>

            <div class="results-card" id="results">
                <h3 class="mb-4" id="resultsTitle"></h3>
                <div id="llmMessage"></div>
                <div id="recommendationsList"></div>
            </div>
        </div>

        <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
        <script>
            function showLoading() {
                document.getElementById('loading').style.display = 'block';
                document.getElementById('results').style.display = 'none';
            }

            function hideLoading() {
                document.getElementById('loading').style.display = 'none';
                document.getElementById('results').style.display = 'block';
            }

            function displayRecommendations(data, isPersonalized) {
                const title = isPersonalized ?
                    `<i class="fas fa-user-check"></i> Recommendations for ${data.user_id || 'You'}` :
                    '<i class="fas fa-fire"></i> Globally Popular Content';

                document.getElementById('resultsTitle').innerHTML = title;

                // Display LLM message if available
                const llmDiv = document.getElementById('llmMessage');
                if (data.llm_message) {
                    llmDiv.innerHTML = `
                        <div class="llm-message">
                            <i class="fas fa-robot"></i> <strong>AI Insight:</strong><br>
                            ${data.llm_message}
                        </div>
                    `;
                } else {
                    llmDiv.innerHTML = '';
                }

                // Display fallback message if used
                if (data.fallback_used) {
                    llmDiv.innerHTML += `
                        <div class="alert alert-info">
                            <i class="fas fa-info-circle"></i> No viewing history found. Showing popular recommendations instead.
                        </div>
                    `;
                }

                const listDiv = document.getElementById('recommendationsList');
                listDiv.innerHTML = '';

                data.items.forEach((item, index) => {
                    const contentTypeColors = {
                        'series': '#3182ce',
                        'movie': '#e53e3e',
                        'tv': '#38a169',
                        'microdrama': '#d69e2e',
                        'anime': '#805ad5'
                    };

                    const genreColors = {
                        'romance': '#ed64a6',
                        'action': '#e53e3e',
                        'drama': '#3182ce',
                        'comedy': '#ecc94b',
                        'thriller': '#2d3748',
                        'family': '#38a169',
                        'kids': '#4299e1',
                        'anime': '#805ad5'
                    };

                    const itemHtml = `
                        <div class="recommendation-item">
                            <div class="item-title">
                                ${index + 1}. ${item.title}
                            </div>
                            <div class="item-meta">
                                <span class="badge badge-custom" style="background-color: ${contentTypeColors[item.content_type] || '#718096'}">
                                    ${item.content_type.toUpperCase()}
                                </span>
                                <span class="badge badge-custom" style="background-color: ${genreColors[item.genre] || '#718096'}">
                                    ${item.genre.toUpperCase()}
                                </span>
                                <span class="badge badge-custom bg-secondary">
                                    Score: ${item.score}
                                </span>
                            </div>
                            <div class="item-reason">
                                <i class="fas fa-info-circle"></i> ${item.reason}
                            </div>
                        </div>
                    `;
                    listDiv.innerHTML += itemHtml;
                });

                hideLoading();
            }

            async function getUserRecommendations() {
                const userId = document.getElementById('userId').value.trim();
                const k = document.getElementById('numRecs').value || 10;

                if (!userId) {
                    alert('Please enter a User ID');
                    return;
                }

                showLoading();

                try {
                    const response = await fetch(`/recommendations?user_id=${userId}&k=${k}`);
                    const data = await response.json();

                    if (response.ok) {
                        displayRecommendations(data, true);
                    } else {
                        alert('Error: ' + data.detail);
                        hideLoading();
                    }
                } catch (error) {
                    alert('Error fetching recommendations: ' + error);
                    hideLoading();
                }
            }

            async function getPopularRecommendations() {
                const k = document.getElementById('numRecs').value || 10;

                showLoading();

                try {
                    const response = await fetch(`/popular?k=${k}`);
                    const data = await response.json();

                    if (response.ok) {
                        displayRecommendations(data, false);
                    } else {
                        alert('Error: ' + data.detail);
                        hideLoading();
                    }
                } catch (error) {
                    alert('Error fetching popular items: ' + error);
                    hideLoading();
                }
            }

            // Allow Enter key to trigger search
            document.getElementById('userId').addEventListener('keypress', function(e) {
                if (e.key === 'Enter') {
                    getUserRecommendations();
                }
            });
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok", "message": "Recommendation system is running"}


@app.get("/popular")
async def get_popular(k: int = Query(default=10, ge=1, le=50)):
    """Get globally popular recommendations"""
    try:
        recommendations = recommend_popular(k)

        return {
            "k": k,
            "items": recommendations,
            "total_items": len(recommendations)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/recommendations")
async def get_recommendations(
    user_id: str = Query(..., description="User ID"),
    k: int = Query(default=10, ge=1, le=50, description="Number of recommendations")
):
    """Get personalized recommendations for a user"""
    try:
        recommendations, fallback_used = recommend_for_user(user_id, k)

        # Try to enhance with LLM
        llm_message = None
        if not fallback_used:
            try:
                llm_message = enhance_recommendations_with_llm(user_id, recommendations)
            except:
                pass

        return {
            "user_id": user_id,
            "k": k,
            "items": recommendations,
            "fallback_used": fallback_used,
            "llm_message": llm_message,
            "total_recommendations": len(recommendations)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/user_history")
async def get_user_history(user_id: str = Query(..., description="User ID")):
    """Get user's watch history"""
    try:
        if user_id not in users_df['user_id'].values:
            raise HTTPException(status_code=404, detail="User not found")

        user_events = events_df[events_df['user_id'] == user_id].merge(
            items_df, on='item_id', how='left'
        )

        user_events = user_events.sort_values('watch_seconds', ascending=False)

        history = []
        for _, row in user_events.head(20).iterrows():
            history.append({
                'item_id': row['item_id'],
                'title': row['title'],
                'content_type': row['content_type'],
                'genre': row['genre'],
                'watch_seconds': int(row['watch_seconds']),
                'event_type': row['event_type'],
                'timestamp': row['timestamp']
            })

        return {
            "user_id": user_id,
            "total_events": len(user_events),
            "history": history
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== RUN APPLICATION ====================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)
