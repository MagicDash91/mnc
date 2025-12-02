"""
Validate and explain recommendation score calculation for a specific item
Shows step-by-step how collaborative filtering calculates the score
"""

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

def validate_score_calculation(user_id='u1', target_item_title='Blue Lock'):
    """
    Show detailed calculation of how a recommendation score is computed
    """

    print("=" * 80)
    print(f"RECOMMENDATION SCORE CALCULATION BREAKDOWN")
    print(f"User: {user_id} | Target Item: {target_item_title}")
    print("=" * 80)

    # Load datasets
    items_df = pd.read_csv('items.csv')
    events_df = pd.read_csv('events.csv')

    # Find target item
    target_items = items_df[items_df['title'].str.contains(target_item_title, case=False, na=False)]

    if len(target_items) == 0:
        print(f"✗ Item '{target_item_title}' not found!")
        return

    target_item_id = target_items.iloc[0]['item_id']
    target_item_full = target_items.iloc[0]['title']

    print(f"\n✓ Found target item: {target_item_full} (ID: {target_item_id})")

    # Build recommendation system
    print("\n" + "=" * 80)
    print("STEP 1: BUILD USER-ITEM INTERACTION MATRIX")
    print("=" * 80)

    event_weights = {
        'play': 1.0,
        'complete': 3.0,
        'like': 2.5,
        'save': 2.0,
        'pause': 0.5,
        'skip': 0.1
    }

    print("\nEvent weights:")
    for event, weight in event_weights.items():
        print(f"  {event}: {weight}x")

    # Calculate interaction scores
    events_df['interaction_score'] = events_df.apply(
        lambda row: row['watch_seconds'] * event_weights.get(row['event_type'], 1.0),
        axis=1
    )

    # Aggregate multiple interactions
    interaction_df = events_df.groupby(['user_id', 'item_id'])['interaction_score'].sum().reset_index()

    # Create user-item matrix
    user_item_matrix = interaction_df.pivot_table(
        index='user_id',
        columns='item_id',
        values='interaction_score',
        fill_value=0
    )

    print(f"\n✓ Created user-item matrix: {user_item_matrix.shape[0]} users × {user_item_matrix.shape[1]} items")

    # Build item similarity matrix
    print("\n" + "=" * 80)
    print("STEP 2: CALCULATE ITEM-ITEM SIMILARITY (COSINE SIMILARITY)")
    print("=" * 80)

    normalized_matrix = normalize(user_item_matrix.values, axis=0)
    item_similarity_matrix = cosine_similarity(normalized_matrix.T)

    item_similarity_df = pd.DataFrame(
        item_similarity_matrix,
        index=user_item_matrix.columns,
        columns=user_item_matrix.columns
    )

    print(f"\n✓ Built item similarity matrix: {item_similarity_df.shape[0]} items × {item_similarity_df.shape[1]} items")

    # Check if user exists
    if user_id not in user_item_matrix.index:
        print(f"\n✗ User {user_id} not found in matrix!")
        return

    # Get user interactions
    print("\n" + "=" * 80)
    print(f"STEP 3: GET USER {user_id}'s WATCHED ITEMS")
    print("=" * 80)

    user_interactions = user_item_matrix.loc[user_id]
    watched_items = user_interactions[user_interactions > user_interactions.quantile(0.7)].index.tolist()

    print(f"\n✓ User {user_id} watched {len(watched_items)} items significantly (top 30% threshold)")
    print("\nItems watched by user:")
    print(f"{'Item ID':<10} {'Title':<40} {'Score':<12}")
    print("-" * 80)

    watched_details = []
    for item_id in watched_items:
        item_info = items_df[items_df['item_id'] == item_id].iloc[0]
        score = user_interactions[item_id]
        watched_details.append({
            'item_id': item_id,
            'title': item_info['title'],
            'score': score
        })
        print(f"{item_id:<10} {item_info['title'][:39]:<40} {score:<12.2f}")

    # Check if target item is already watched
    if target_item_id in watched_items:
        print(f"\n⚠️  Note: {target_item_full} is already in watched items!")
        print("   The recommendation system would NOT recommend this item.")
        print("   Continuing calculation for demonstration...\n")

    # Calculate recommendation score
    print("\n" + "=" * 80)
    print(f"STEP 4: CALCULATE RECOMMENDATION SCORE FOR '{target_item_full}'")
    print("=" * 80)

    if target_item_id not in item_similarity_df.index:
        print(f"\n✗ Target item {target_item_id} not in similarity matrix!")
        return

    print(f"\nFormula: Score = Σ (similarity × user_interaction_weight)")
    print(f"For each watched item, we multiply:")
    print(f"  - How similar it is to {target_item_full}")
    print(f"  - How much user {user_id} liked that item")

    total_score = 0
    print(f"\n{'Watched Item':<40} {'Similarity':<12} {'User Score':<12} {'Contribution':<15}")
    print("-" * 80)

    contributions = []
    for watched_item_id in watched_items:
        if watched_item_id in item_similarity_df.columns:
            similarity = item_similarity_df.loc[target_item_id, watched_item_id]
            user_score = user_interactions[watched_item_id]
            contribution = similarity * user_score
            total_score += contribution

            watched_item_title = items_df[items_df['item_id'] == watched_item_id].iloc[0]['title']

            contributions.append({
                'item': watched_item_title,
                'similarity': similarity,
                'user_score': user_score,
                'contribution': contribution
            })

            print(f"{watched_item_title[:39]:<40} {similarity:<12.4f} {user_score:<12.2f} {contribution:<15.2f}")

    print("-" * 80)
    print(f"{'TOTAL RECOMMENDATION SCORE':<40} {'':12} {'':12} {total_score:<15.4f}")

    # Show top contributors
    print("\n" + "=" * 80)
    print("TOP 5 CONTRIBUTORS TO THE SCORE")
    print("=" * 80)

    contributions_sorted = sorted(contributions, key=lambda x: x['contribution'], reverse=True)
    print(f"\n{'Rank':<6} {'Item':<40} {'Contribution':<15} {'%':<10}")
    print("-" * 80)

    for idx, contrib in enumerate(contributions_sorted[:5], 1):
        percentage = (contrib['contribution'] / total_score * 100) if total_score > 0 else 0
        print(f"{idx:<6} {contrib['item'][:39]:<40} {contrib['contribution']:<15.2f} {percentage:<10.1f}%")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\nTarget Item: {target_item_full}")
    print(f"User: {user_id}")
    print(f"Final Recommendation Score: {total_score:.4f}")
    print(f"\nThis score is calculated by:")
    print(f"  1. Finding all items user {user_id} watched significantly")
    print(f"  2. Measuring similarity between each watched item and {target_item_full}")
    print(f"  3. Weighting by how much the user liked each watched item")
    print(f"  4. Summing all weighted similarities")

    if total_score > 0:
        print(f"\n✓ Higher score = Better recommendation")
    else:
        print(f"\n✗ Score of 0 = No similarity found with watched items")

    # Export detailed breakdown
    breakdown_df = pd.DataFrame(contributions_sorted)
    output_file = f'score_breakdown_{user_id}_{target_item_id}.csv'
    breakdown_df.to_csv(output_file, index=False)
    print(f"\n✓ Detailed breakdown exported to: {output_file}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    import sys

    # Default values
    user_id = 'u1'
    target_item = 'Blue Lock'

    # Check command line arguments
    if len(sys.argv) > 1:
        user_id = sys.argv[1]
    if len(sys.argv) > 2:
        target_item = sys.argv[2]

    print(f"\nUsage: python validate_score_calculation.py [user_id] [item_name]")
    print(f"Example: python validate_score_calculation.py u1 'Blue Lock'")
    print(f"\nUsing: user_id={user_id}, item={target_item}\n")

    validate_score_calculation(user_id, target_item)
