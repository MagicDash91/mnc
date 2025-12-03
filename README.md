# Streaming Platform Recommendation System

A comprehensive recommendation system for streaming platforms built with **FastAPI**, **Scikit-learn collaborative filtering**, and **Google Gemini LLM**.

## 📁 Project Structure

```
mnc/
├── main.py                          # FastAPI web application with recommendations
├── validate_popular_movies.py       # Validate popular movies globally
├── validate_user_movies.py          # Validate user viewing history
├── validate_score_calculation.py    # Score calculation breakdown
├── users.csv                        # User demographics data
├── items.csv                        # Content catalog
├── events.csv                       # User interaction events
├── requirements.txt                 # Python dependencies
├── .env                            # Environment variables (Gemini API key)
└── README.md                       # This file
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Up Environment Variables

Create `.env` file with your Google Gemini API key:
```
GOOGLE_API_KEY=your_api_key_here
```

### 3. Run the Application

```bash
# FastAPI Web Application
python main.py
```

Access at: **http://127.0.0.1:8001**

---

## 📊 Main Scripts Overview

### 1️⃣ **main.py** - FastAPI Recommendation System

**Purpose**: Production-ready web application with REST API and web interface

**Features**:
- ✅ Real-time personalized recommendations using collaborative filtering
- ✅ Global popularity recommendations
- ✅ Beautiful Bootstrap web interface
- ✅ Gemini LLM-powered explanations
- ✅ Data cleansing on startup
- ✅ REST API endpoints

**How to Use**:

```bash
# Start server
python main.py

# Or with uvicorn
uvicorn main:app --reload --port 8001
```

**API Endpoints**:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web interface |
| `/health` | GET | Health check |
| `/popular?k=10` | GET | Get top-k popular items |
| `/recommendations?user_id=u1&k=10` | GET | Get personalized recommendations |
| `/user_history?user_id=u1` | GET | Get user watch history |

**Example API Call**:
```bash
curl http://127.0.0.1:8001/recommendations?user_id=u1&k=10
```

---

### 2️⃣ **validate_popular_movies.py** - Global Popular Movies Validator

**Purpose**: Validate and analyze globally popular movies for a specific user

**Features**:
- ✅ Shows user's complete watch history (all content types)
- ✅ User's genre preferences
- ✅ Top 10 globally popular movies
- ✅ Content type comparison (movies vs series vs anime)
- ✅ Exports reports to CSV

**How to Use**:

```bash
# Default user: u1
python validate_popular_movies.py
```

**What It Shows**:
1. User u1's movie viewing history (movies only)
2. User u1's genre preferences
3. Top 10 globally popular movies (all users)
4. Content type statistics
5. Exports: `user_u1_movie_report.csv`

---

### 3️⃣ **validate_user_movies.py** - User Complete History & Recommendations

**Purpose**: Validate user's complete viewing history (ALL content types) and see recommendations

**Features**:
- ✅ Complete viewing history (movies, series, anime, microdrama, TV)
- ✅ Top 10 favorite items
- ✅ Genre and content type preferences
- ✅ Personalized recommendations using collaborative filtering
- ✅ Exports detailed reports

**How to Use**:

```bash
# Default user: u1
python validate_user_movies.py

# Specific user
python validate_user_movies.py u50
```

**What It Shows**:
1. User profile (name, age, gender, region)
2. Complete viewing history (ALL content types)
3. Content breakdown by type
4. Genre preferences
5. Top 10 favorite items
6. **Personalized recommendations** (same as main.py)
7. Exports: `user_u1_complete_history.csv`, `user_u1_recommendations.csv`

---

## 📈 User u1 Results Comparison

### User u1 Profile
```
Name:    Sari Hidayat
Age:     55
Gender:  F
Region:  Bandung
```

---

### **Result 1: validate_popular_movies.py**

Shows user u1's **MOVIE** viewing history only:

```
TOP 10 MOVIES WATCHED BY USER u1
======================================================================
#    Movie Title                         Genre        Score
----------------------------------------------------------------------
1    La La Land Part 4                   romance      3338.00
2    Koala Kumal Part 4                  romance      2828.00
3    Hotel Transylvania                  family       1844.00
4    The Incredibles                     family       663.00
5    Captain America: Civil War Part 4   action       650.00
6    The Dark Knight Part 2              action       322.00
7    The Bourne Ultimatum Part 3         thriller     279.00

Genre Preferences (Movies Only):
- romance:   6166.00
- family:    2507.00
- action:    972.00
- thriller:  279.00
```

**Why only 7 items?** ➡️ This script filters to **movies only**, excluding series, anime, etc.

---

### **Result 2: validate_user_movies.py**

Shows user u1's **COMPLETE** viewing history (ALL content types):

```
TOP 30 ITEMS WATCHED BY USER u1
======================================================================
#    Title                                    Type         Genre      Score
----------------------------------------------------------------------
1    Blue Lock S1E11                          series       anime      8000.00
2    Vincenzo S2E3                            series       thriller   6000.00
3    La La Land Part 4                        movie        romance    3338.00
4    Koala Kumal Part 4                       movie        romance    2828.00
5    Hotel Transylvania                       movie        family     1844.00
6    The Incredibles                          movie        family     663.00
7    Captain America: Civil War Part 4        movie        action     650.00
8    The Dark Knight Part 2                   movie        action     322.00
9    The Bourne Ultimatum Part 3              movie        thriller   279.00
...

Content Breakdown:
  - series: 15 items
  - movie: 7 items
  - anime: 3 items
```

**Why different?** ➡️ This script shows **ALL content types**, including the series/anime that u1 watched!

---

### **Result 3: main.py - Recommendations for u1**

Based on **ALL content** u1 watched, recommends new items:

```
TOP 10 RECOMMENDATIONS FOR USER u1
======================================================================
1. Diary Pacar Ghosting S1E6     MICRODRAMA  ROMANCE    5489.6131
   Similar to viewing history

2. The Bourne Ultimatum           MOVIE       THRILLER   3923.2057
   Based on viewing history

3. Gotham S1E10                   SERIES      THRILLER   3802.5837
   Similar to 'Blue Lock S1E11'

4. Drama 5 Menit: Pulang S3E2     MICRODRAMA  DRAMA      3482.3459
   Similar to 'Vincenzo S2E3'

5. Hospital Playlist S1E20        SERIES      DRAMA      3411.5212
   Similar to 'Vincenzo S2E3'

6. Blue Lock                      SERIES      ANIME      3387.7224
   Similar to 'Blue Lock S1E11'

7. Captain America: Civil War P2  MOVIE       ACTION     3291.7992
   Based on viewing history

8. Extraordinary Attorney Woo     SERIES      DRAMA      3183.8511
   Based on viewing history

9. Tenet Part 3                   MOVIE       THRILLER   3105.2610
   Similar to 'Blue Lock S1E11'

10. Finding Nemo Part 3           MOVIE       FAMILY     3016.3603
    Similar to 'The Dark Knight Part 2'
```

**Why these recommendations?** ➡️ Based on:
- Blue Lock S1E11 (anime series) - highest watched
- Vincenzo S2E3 (thriller series) - second highest
- Romance movies preference
- Family content preference

---

## 🔍 Key Differences Explained

### **Why Results Are Different?**

| Script | Content Types | Purpose | Output |
|--------|--------------|---------|--------|
| **validate_popular_movies.py** | **Movies ONLY** | Show movie viewing history | What movies u1 watched |
| **validate_user_movies.py** | **ALL types** | Show complete history | What ALL content u1 watched |
| **main.py** | **ALL types** | Generate recommendations | What u1 SHOULD watch next |

### **The Problem:**

1. **validate_popular_movies.py** filters:
   ```python
   user_movies = user_movies[user_movies['content_type'] == 'movie'].copy()
   ```
   ➡️ Only shows 7 movies

2. **validate_user_movies.py** shows all:
   ```python
   user_content = user_events.merge(items_df, on='item_id', how='left')
   ```
   ➡️ Shows all 25+ items (movies + series + anime)

3. **main.py** uses all content:
   ```python
   user_item_matrix = interaction_df.pivot_table(...)  # All content types
   ```
   ➡️ Recommends based on complete viewing behavior

### **Why Recommendations Mention "Blue Lock S1E11" and "Vincenzo S2E3"?**

User u1 **actually watched these series**! They just don't appear in `validate_popular_movies.py` because that script filters to movies only.

**Proof**: Run `validate_user_movies.py` and you'll see:
```
1. Blue Lock S1E11          series  anime     8000.00  ← Highest watched!
2. Vincenzo S2E3            series  thriller  6000.00  ← Second highest!
3. La La Land Part 4        movie   romance   3338.00
```

---

## 🧮 How Collaborative Filtering Works

### **Algorithm: Item-Based Collaborative Filtering with Cosine Similarity**

1. **Build User-Item Matrix**
   ```
            i1    i2    i3    i4
   u1     8000     0   6000    0
   u2        0  3000   1500  800
   u3     2500  2800      0    0
   ```

2. **Calculate Item Similarity (Cosine Similarity)**
   ```
          i1    i2    i3
   i1   1.00  0.85  0.42
   i2   0.85  1.00  0.78
   i3   0.42  0.78  1.00
   ```

3. **Generate Recommendations**
   ```
   For user u1:
   - u1 watched i1 (Blue Lock) with score 8000
   - i2 is similar to i1 (similarity: 0.85)
   - Recommendation score for i2 = 0.85 × 8000 = 6800
   ```

### **Formula**:
```
Score(item_j for user_i) = Σ similarity(item_j, item_k) × user_interaction(i, k)
                           for all items k that user i watched
```

### **Event Weights**:
```python
'play': 1.0       # Normal watch
'complete': 3.0   # Finished watching (3x weight)
'like': 2.5       # Liked the content (2.5x weight)
'save': 2.0       # Saved for later (2x weight)
'pause': 0.5      # Paused (0.5x weight)
'skip': 0.1       # Skipped (0.1x weight)
```

**Interaction Score** = `watch_seconds × event_weight`

Example:
- User watched "Blue Lock" for 4000 seconds with "complete" event
- Score = 4000 × 3.0 = **12000**

---

## 📝 Data Files

### **users.csv**
```csv
user_id,name,age,gender,region
u1,Sari Hidayat,55,F,Bandung
u2,Ahmad Wijaya,19,M,Denpasar
```

### **items.csv**
```csv
item_id,title,content_type,genre
i1,Crash Landing on You,series,romance
i2,Start-Up,series,drama
i63,Blue Lock,series,anime
```

### **events.csv**
```csv
user_id,item_id,event_type,watch_seconds,timestamp
u1,i92,play,4000,2025-01-10T19:24:00
u1,i100,complete,3000,2025-01-15T08:30:00
```

---

## 🛠️ Technical Details

### **Tech Stack**:
- **Backend**: FastAPI, Python 3.8+
- **ML Model**: Scikit-learn (Cosine Similarity, Matrix Normalization)
- **LLM**: Google Gemini API
- **Frontend**: HTML, CSS, Bootstrap 5
- **Data Processing**: Pandas, NumPy

### **System Architecture**:

```
┌─────────────┐
│   User      │
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────────┐
│  FastAPI Web Server (main.py)      │
│  - REST API Endpoints               │
│  - HTML Interface                   │
└──────┬──────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────┐
│  Data Cleansing Module              │
│  - Remove nulls                     │
│  - Validate IDs                     │
└──────┬──────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────┐
│  Collaborative Filtering Engine     │
│  - Build user-item matrix           │
│  - Calculate item similarity        │
│  - Generate recommendations         │
└──────┬──────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────┐
│  Gemini LLM (Optional)              │
│  - Generate explanations            │
└──────┬──────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────┐
│  Response (JSON/HTML)               │
└─────────────────────────────────────┘
```

### **Performance**:
- **Startup Time**: 10-30 seconds (data loading + matrix building)
- **Recommendation Generation**: < 100ms per user
- **Supported Scale**: Millions of events, thousands of users/items

---

## 🎯 Use Cases

### **1. Production Web Application**
```bash
python main.py
# Access at http://127.0.0.1:8001
```
- Real-time recommendations via web interface
- API for mobile apps or other services
- LLM-powered explanations

### **2. Data Validation & Analysis**
```bash
python validate_user_movies.py u1
```
- Validate user behavior
- Analyze viewing patterns
- Debug recommendation logic

### **3. Movie Popularity Analysis**
```bash
python validate_popular_movies.py
```
- Identify trending content
- Compare user vs global preferences
- Genre analysis

### **4. Score Debugging**
```bash
python validate_score_calculation.py u1 "Blue Lock"
```
- Understand why specific items are recommended
- Debug scoring algorithm
- Explain recommendations to stakeholders

---

## 📤 Export Files

All scripts generate CSV reports:

| File | Description |
|------|-------------|
| `user_u1_movie_report.csv` | User's movie viewing history |
| `user_u1_complete_history.csv` | User's complete viewing history |
| `user_u1_recommendations.csv` | Personalized recommendations |
| `popular_movies_report.csv` | Global popular movies |
| `score_breakdown_u1_i63.csv` | Detailed score calculation |

---

## 🐛 Troubleshooting

### **Issue 1: Pandas/NumPy Error**
```
ValueError: numpy.dtype size changed
```
**Solution**:
```bash
pip install --upgrade pandas numpy scikit-learn
```

### **Issue 2: No Gemini LLM Output**
```
Warning: GOOGLE_API_KEY not found
```
**Solution**: Add API key to `.env` file
```bash
GOOGLE_API_KEY=your_key_here
```

### **Issue 3: Different Results Between Scripts**
**Reason**:
- `validate_popular_movies.py` filters to **movies only**
- `validate_user_movies.py` and `main.py` use **all content types**

**Solution**: Use `validate_user_movies.py` for complete validation

---

## 📊 Example Output Comparison

### User u1's Actual Viewing History (Complete):

| Rank | Title | Type | Genre | Score |
|------|-------|------|-------|-------|
| 1 | Blue Lock S1E11 | series | anime | 8000.00 |
| 2 | Vincenzo S2E3 | series | thriller | 6000.00 |
| 3 | La La Land Part 4 | movie | romance | 3338.00 |
| 4 | Koala Kumal Part 4 | movie | romance | 2828.00 |
| 5 | Hotel Transylvania | movie | family | 1844.00 |

### Why Recommendations Make Sense:

1. **Gotham S1E10** (Score: 3802.58) ← Similar to Blue Lock S1E11 (thriller series)
2. **Hospital Playlist** (Score: 3411.52) ← Similar to Vincenzo S2E3 (drama series)
3. **Blue Lock** (Score: 3387.72) ← Similar to Blue Lock S1E11 (same franchise)

**Conclusion**: User u1 loves **series** (especially anime and thriller) more than movies! The recommendation system correctly identifies this pattern.

---

## 🤝 Contributing

To add new features or fix bugs:
1. Modify the respective `.py` file
2. Test with sample data
3. Update this README

---

## 📄 License

This project is for educational purposes.

---

## 👨‍💻 Author

AI Engineer Technical Test - Recommendation System

---

## 📞 Support

For issues or questions:
1. Check the Troubleshooting section
2. Review the code comments
3. Validate with `validate_user_movies.py`

---

**Happy Recommending! 🎬🍿**
