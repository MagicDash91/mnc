"""
Script to validate the most popular movie titles
Based on items.csv and events.csv data
Default user: u1
"""

import pandas as pd

def get_user_movie_history(user_id='u1'):
    """Get a specific user's movie viewing history"""

    print("\n" + "=" * 70)
    print(f"USER {user_id} - MOVIE VIEWING HISTORY")
    print("=" * 70)

    # Load datasets
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

    # Get user's events for movies only
    user_events = events_df[events_df['user_id'] == user_id].copy()

    # Merge with items to get movie details
    user_movies = user_events.merge(items_df, on='item_id', how='left')
    user_movies = user_movies[user_movies['content_type'] == 'movie'].copy()

    if len(user_movies) == 0:
        print(f"\n✗ User {user_id} has not watched any movies yet!")
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
    user_movies['engagement_score'] = user_movies.apply(
        lambda row: row['watch_seconds'] * event_weights.get(row['event_type'], 1.0),
        axis=1
    )

    # Sort by engagement score
    user_movies = user_movies.sort_values('engagement_score', ascending=False)

    print(f"\n✓ Found {len(user_movies)} movie interactions")
    print(f"✓ Total watch time: {user_movies['watch_seconds'].sum()} seconds ({user_movies['watch_seconds'].sum()/3600:.1f} hours)")

    # Display watched movies
    print("\n" + "=" * 70)
    print(f"TOP 10 MOVIES WATCHED BY USER {user_id}")
    print("=" * 70)
    print(f"{'#':<4} {'Movie Title':<35} {'Genre':<12} {'Event':<10} {'Watch(s)':<10} {'Score':<10}")
    print("-" * 70)

    for idx, (_, row) in enumerate(user_movies.head(10).iterrows(), 1):
        print(f"{idx:<4} {row['title'][:34]:<35} {row['genre']:<12} {row['event_type']:<10} {int(row['watch_seconds']):<10} {row['engagement_score']:.2f}")

    # Genre preferences
    print("\n" + "=" * 70)
    print(f"USER {user_id} - GENRE PREFERENCES")
    print("=" * 70)

    genre_stats = user_movies.groupby('genre').agg({
        'item_id': 'count',
        'watch_seconds': 'sum',
        'engagement_score': 'sum'
    }).sort_values('engagement_score', ascending=False)

    print(f"{'Genre':<15} {'Movies':<10} {'Watch Time(s)':<15} {'Engagement':<15}")
    print("-" * 70)
    for genre, row in genre_stats.iterrows():
        print(f"{genre:<15} {int(row['item_id']):<10} {int(row['watch_seconds']):<15} {row['engagement_score']:.2f}")

    # Export user report
    output_file = f'user_{user_id}_movie_report.csv'
    user_movies.to_csv(output_file, index=False)
    print(f"\n✓ User report exported to: {output_file}")

    return user_movies


def find_most_popular_movies(top_n=10):
    """Find the most popular movies based on user interactions"""

    print("=" * 70)
    print(f"TOP {top_n} MOST POPULAR MOVIES VALIDATION")
    print("=" * 70)

    # Load datasets
    print("\nLoading data...")
    items_df = pd.read_csv('items.csv')
    events_df = pd.read_csv('events.csv')

    print(f"✓ Loaded {len(items_df)} items")
    print(f"✓ Loaded {len(events_df)} events")

    # Filter for movies only
    movies_df = items_df[items_df['content_type'] == 'movie'].copy()
    print(f"\n✓ Found {len(movies_df)} movies in catalog")

    # Weight different event types
    event_weights = {
        'play': 1.0,
        'complete': 3.0,
        'like': 2.5,
        'save': 2.0,
        'pause': 0.5,
        'skip': 0.1
    }

    # Calculate weighted score for each event
    events_df['weighted_score'] = events_df.apply(
        lambda row: row['watch_seconds'] * event_weights.get(row['event_type'], 1.0),
        axis=1
    )

    # Aggregate by item
    popularity = events_df.groupby('item_id').agg({
        'weighted_score': 'sum',
        'user_id': 'nunique',  # unique users
        'event_type': 'count',  # total events
        'watch_seconds': 'sum'
    }).reset_index()

    popularity.columns = ['item_id', 'total_score', 'unique_users', 'total_events', 'total_watch_seconds']

    # Merge with items to get movie details
    movie_popularity = popularity.merge(movies_df, on='item_id', how='inner')

    # Calculate final popularity score
    if len(movie_popularity) > 0:
        movie_popularity['popularity_score'] = (
            0.5 * (movie_popularity['total_score'] / movie_popularity['total_score'].max()) +
            0.3 * (movie_popularity['unique_users'] / movie_popularity['unique_users'].max()) +
            0.2 * (movie_popularity['total_events'] / movie_popularity['total_events'].max())
        )

    # Sort by popularity
    movie_popularity = movie_popularity.sort_values('popularity_score', ascending=False)

    # Display results
    print("\n" + "=" * 70)
    print(f"TOP {top_n} MOST POPULAR MOVIES")
    print("=" * 70)
    print(f"{'Rank':<6} {'Movie Title':<35} {'Genre':<12} {'Score':<8}")
    print("-" * 70)

    for idx, row in movie_popularity.head(top_n).iterrows():
        rank = list(movie_popularity.index).index(idx) + 1
        print(f"{rank:<6} {row['title'][:34]:<35} {row['genre']:<12} {row['popularity_score']:.4f}")

    # Detailed stats for top movie
    print("\n" + "=" * 70)
    print("DETAILED STATS FOR #1 MOVIE")
    print("=" * 70)

    if len(movie_popularity) > 0:
        top_movie = movie_popularity.iloc[0]
        print(f"Title:              {top_movie['title']}")
        print(f"Item ID:            {top_movie['item_id']}")
        print(f"Genre:              {top_movie['genre']}")
        print(f"Popularity Score:   {top_movie['popularity_score']:.4f}")
        print(f"Unique Viewers:     {int(top_movie['unique_users'])} users")
        print(f"Total Events:       {int(top_movie['total_events'])} interactions")
        print(f"Total Watch Time:   {int(top_movie['total_watch_seconds'])} seconds ({int(top_movie['total_watch_seconds']/3600):.1f} hours)")
        print(f"Weighted Score:     {top_movie['total_score']:.2f}")

    # Show genre distribution
    print("\n" + "=" * 70)
    print("POPULAR MOVIES BY GENRE")
    print("=" * 70)

    genre_stats = movie_popularity.groupby('genre').agg({
        'item_id': 'count',
        'popularity_score': 'mean'
    }).sort_values('popularity_score', ascending=False)

    print(f"{'Genre':<15} {'Count':<10} {'Avg Popularity':<15}")
    print("-" * 70)
    for genre, row in genre_stats.iterrows():
        print(f"{genre:<15} {int(row['item_id']):<10} {row['popularity_score']:.4f}")

    # Export to CSV
    output_file = 'popular_movies_report.csv'
    movie_popularity.to_csv(output_file, index=False)
    print(f"\n✓ Full report exported to: {output_file}")

    return movie_popularity


def compare_with_all_content():
    """Compare movie popularity with all content types"""

    print("\n" + "=" * 70)
    print("MOVIES VS OTHER CONTENT TYPES")
    print("=" * 70)

    # Load datasets
    items_df = pd.read_csv('items.csv')
    events_df = pd.read_csv('events.csv')

    # Weight different event types
    event_weights = {
        'play': 1.0,
        'complete': 3.0,
        'like': 2.5,
        'save': 2.0,
        'pause': 0.5,
        'skip': 0.1
    }

    events_df['weighted_score'] = events_df.apply(
        lambda row: row['watch_seconds'] * event_weights.get(row['event_type'], 1.0),
        axis=1
    )

    # Aggregate by item
    popularity = events_df.groupby('item_id').agg({
        'weighted_score': 'sum',
        'user_id': 'nunique',
        'event_type': 'count'
    }).reset_index()

    popularity.columns = ['item_id', 'total_score', 'unique_users', 'total_events']

    # Merge with items
    content_popularity = popularity.merge(items_df, on='item_id', how='left')

    # Calculate popularity score
    if len(content_popularity) > 0:
        content_popularity['popularity_score'] = (
            0.5 * (content_popularity['total_score'] / content_popularity['total_score'].max()) +
            0.3 * (content_popularity['unique_users'] / content_popularity['unique_users'].max()) +
            0.2 * (content_popularity['total_events'] / content_popularity['total_events'].max())
        )

    # Group by content type
    content_type_stats = content_popularity.groupby('content_type').agg({
        'item_id': 'count',
        'popularity_score': ['mean', 'max'],
        'unique_users': 'sum',
        'total_events': 'sum'
    }).round(4)

    print(f"\n{'Content Type':<15} {'Items':<10} {'Avg Score':<12} {'Max Score':<12} {'Total Users':<15}")
    print("-" * 70)
    for content_type, row in content_type_stats.iterrows():
        print(f"{content_type:<15} {int(row[('item_id', 'count')]):<10} {row[('popularity_score', 'mean')]:<12.4f} {row[('popularity_score', 'max')]:<12.4f} {int(row[('unique_users', 'sum')]):<15}")

    # Show top 5 across all content
    print("\n" + "=" * 70)
    print("TOP 5 ITEMS ACROSS ALL CONTENT TYPES")
    print("=" * 70)

    top_all = content_popularity.nlargest(5, 'popularity_score')
    print(f"{'Rank':<6} {'Title':<30} {'Type':<12} {'Genre':<12} {'Score':<8}")
    print("-" * 70)

    for rank, (idx, row) in enumerate(top_all.iterrows(), 1):
        print(f"{rank:<6} {row['title'][:29]:<30} {row['content_type']:<12} {row['genre']:<12} {row['popularity_score']:.4f}")


if __name__ == "__main__":
    # Set user_id here
    user_id = 'u1'  # Change this to any user ID you want to check

    print("=" * 70)
    print(f"MOVIE VALIDATION FOR USER: {user_id}")
    print("=" * 70)

    # Get user-specific movie history
    user_movies = get_user_movie_history(user_id)

    # Find most popular movies globally
    print("\n" + "=" * 70)
    print("GLOBAL POPULAR MOVIES (FOR COMPARISON)")
    print("=" * 70)
    movie_popularity = find_most_popular_movies(top_n=10)

    # Compare with other content types
    compare_with_all_content()

    print("\n" + "=" * 70)
    print("✓ VALIDATION COMPLETE!")
    print("=" * 70)
