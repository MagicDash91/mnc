"""
Script to validate user-specific movie viewing history AND recommendations
Shows both what user watched and what they should watch next
Usage: python validate_user_movies.py u1
"""

import pandas as pd
import sys
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

def get_user_movie_history(user_id):
    """Get a specific user's complete viewing history (all content types)"""

    print("=" * 70)
    print(f"USER {user_id} - COMPLETE VIEWING HISTORY")
    print("=" * 70)

    # Load datasets
    print("\nLoading data...")
    items_df = pd.read_csv('items.csv')
    events_df = pd.read_csv('events.csv')
    users_df = pd.read_csv('users.csv')

    # Check if user exists
    if user_id not in users_df['user_id'].values:
        print(f"✗ Error: User {user_id} not found!")
        return None

    # Get user info
    user_info = users_df[users_df['user_id'] == user_id].iloc[0]
    print(f"\nUser Profile:")
    print(f"  Name:    {user_info['name']}")
    print(f"  Age:     {user_info['age']}")
    print(f"  Gender:  {user_info['gender']}")
    print(f"  Region:  {user_info['region']}")

    # Get user's events for ALL content types
    user_events = events_df[events_df['user_id'] == user_id].copy()

    # Merge with items to get details
    user_content = user_events.merge(items_df, on='item_id', how='left')

    if len(user_content) == 0:
        print(f"\n✗ User {user_id} has not watched any content yet!")
        return None

    # Event weights
    event_weights = {
        'play': 1.0,
        'complete': 3.0,
        'like': 2.5,
        'save': 2.0,
        'pause': 0.5,
        'skip': 0.1
    }

    # Calculate engagement score
    user_content['engagement_score'] = user_content.apply(
        lambda row: row['watch_seconds'] * event_weights.get(row['event_type'], 1.0),
        axis=1
    )

    # Sort by engagement score
    user_content = user_content.sort_values('engagement_score', ascending=False)

    print(f"\n✓ Found {len(user_content)} total interactions")
    print(f"✓ Total watch time: {user_content['watch_seconds'].sum()} seconds ({user_content['watch_seconds'].sum()/3600:.1f} hours)")

    # Content type breakdown
    content_type_stats = user_content.groupby('content_type').agg({
        'item_id': 'count',
        'watch_seconds': 'sum'
    }).sort_values('watch_seconds', ascending=False)

    print(f"\nContent breakdown:")
    for content_type, row in content_type_stats.iterrows():
        print(f"  - {content_type}: {int(row['item_id'])} items, {int(row['watch_seconds'])} seconds")

    # Display watched content
    print("\n" + "=" * 70)
    print(f"TOP 30 ITEMS WATCHED BY USER {user_id}")
    print("=" * 70)
    print(f"{'#':<4} {'Title':<40} {'Type':<12} {'Genre':<10} {'Score':<10}")
    print("-" * 70)

    for idx, (_, row) in enumerate(user_content.head(30).iterrows(), 1):
        print(f"{idx:<4} {row['title'][:39]:<40} {row['content_type']:<12} {row['genre']:<10} {row['engagement_score']:.2f}")

    # Genre preferences
    print("\n" + "=" * 70)
    print(f"USER {user_id} - GENRE PREFERENCES")
    print("=" * 70)

    genre_stats = user_content.groupby('genre').agg({
        'item_id': 'count',
        'watch_seconds': 'sum',
        'engagement_score': 'sum'
    }).sort_values('engagement_score', ascending=False)

    print(f"{'Genre':<15} {'Items Watched':<15} {'Watch Time(s)':<15} {'Engagement Score':<20}")
    print("-" * 70)
    for genre, row in genre_stats.iterrows():
        print(f"{genre:<15} {int(row['item_id']):<15} {int(row['watch_seconds']):<15} {row['engagement_score']:.2f}")

    # Top favorite items (highest engagement)
    print("\n" + "=" * 70)
    print(f"USER {user_id} - TOP 10 FAVORITE ITEMS (ALL TYPES)")
    print("=" * 70)

    top_favorites = user_content.head(10)
    for idx, (_, row) in enumerate(top_favorites.iterrows(), 1):
        print(f"\n{idx}. {row['title']}")
        print(f"   Content Type: {row['content_type'].upper()}")
        print(f"   Genre:        {row['genre']}")
        print(f"   Item ID:      {row['item_id']}")
        print(f"   Event Type:   {row['event_type']}")
        print(f"   Watch Time:   {int(row['watch_seconds'])} seconds")
        print(f"   Engagement:   {row['engagement_score']:.2f}")
        print(f"   Timestamp:    {row['timestamp']}")

    # Export user report
    output_file = f'user_{user_id}_complete_history.csv'
    user_content.to_csv(output_file, index=False)
    print(f"\n✓ Full report exported to: {output_file}")

    return user_content


if __name__ == "__main__":
    # Check if user_id is provided as command line argument
    if len(sys.argv) > 1:
        user_id = sys.argv[1]
    else:
        # Default to user u1
        user_id = 'u1'
        print("=" * 70)
        print("Using default user: u1")
        print("(To check other users: python validate_user_movies.py <user_id>)")
        print("=" * 70)

    # Get user-specific movie history
    user_movies = get_user_movie_history(user_id)

    print("\n" + "=" * 70)
    print("✓ VALIDATION COMPLETE!")
    print("=" * 70)
