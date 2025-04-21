# Memory Cartography

memory-cartography/
├── frontend/
│   ├── src/
│   │   ├── hooks/
│   │   │   ├── useMemories.js          # Hook for fetching memories
│   │   │   ├── useSearch.js            # Hook for search functionality
│   │   │   └── useNarrative.js         # Hook for narrative generation
│   ├── package.json
│   └── README.md

## Implemetation
1. Scrape data:
```
python backend\scripts\pexels_scraper.py
```
Put all raw pics in: `backend\data\raw\pubic_photos` & `backend\data\raw\user_photos`

2. Process data:
```
python backend\scripts\process_data.py
```

3. Run
```
cd backend
uvicorn app.main:app --reload

cd frontend
npm start
```


### Memory Cartography Implementation Guide

I've made several changes to improve your Memory Cartography application, focusing specifically on the weight-based memory sorting and narrative generation. Here's what's changed and how to implement these improvements:

## Backend Changes

### 1. Enhanced Search Endpoint (`main.py`)

The `search_memories_endpoint` now includes:
- A new `sort_by` parameter that accepts "weight", "date", or "relevance"
- Better sorting functionality that properly respects memory weights
- Retrieving more initial results before sorting and limiting (to ensure you get the best matches)
- Better error handling and logging

### 2. Improved Narrative Generation (`main.py`)

The `generate_narrative_endpoint` now uses:
- Weight-prioritized memory selection - higher weight memories are guaranteed to be included
- A "total weight target" approach that ensures important memories are included
- Chronological reordering of selected memories to create more coherent narratives
- Better logging to track which memories are being used

## Frontend Changes (`MemoryApp.tsx`)

The React component now includes:
- A new sort selector UI with options for weight, date, and relevance
- Client-side re-sorting when weights are modified
- Immediate updates to the UI when weights change
- Enter key support on the search field for better usability
- Empty state handling when no results are found

## How to Implement

1. **Update `main.py`**:
   - Replace the `search_memories_endpoint` with the new version
   - Replace the `generate_narrative_endpoint` with the new version

2. **Update `MemoryApp.tsx`**:
   - Replace your React component with the updated version
   - This adds the sorting UI and improves the memory weight handling

3. **Additional Dependencies**:
   - No new dependencies are required

## Expected Behavior

After implementing these changes:

1. **Search Results**:
   - Memory cards will be sorted by weight by default
   - Users can change sorting to date or relevance as needed
   - Memories with higher weights appear at the top

2. **Weight Adjustments**:
   - When you increase or decrease a memory's weight, it will automatically move to the correct position
   - The visual weight indicator will update immediately

3. **Narrative Generation**:
   - Narratives will prioritize higher-weight memories
   - The system will still ensure a mixture of memories for interesting narratives
   - Important memories will be more likely to appear in narratives
