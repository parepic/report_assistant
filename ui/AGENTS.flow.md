# Report Assistant – Simplified User Flow (MVP v2)

## Entry Point: Landing Page

### Purpose
Immediately communicate:
1. What the app does
2. That it works on *existing reports*
3. A single, clear next action

No exploration, no configuration, no accounts.

This is only the beginning, it does not outline all the features planned for the app.

---

UI Phases:
- landing
- selecting_single_report
- selecting_multi_report
- chat

Chat Options:
- Single Report
- Multi Report
<!-- - Time Comparison -->

---

## Hero Section

- Large headline (centered):
  **“Boost Your Financial Analysis”**

- Supporting text (1–2 lines):
  “Analyze annual reports from leading publicly traded companies using natural language.”

- Three CTAs:
  **"Analyze a Company's Report"**
  **"Compare Multiple Reports"**
  **"Add Your Own Report"**


---

## Data Source
- Reports are loaded **only** from:
```
project_root/data/reports/
```

This directory is assumed to contain a small, curated set of reports.



### Single Report Modal Layout

**Header**
- Title: “Select a report to analyze”

**Content**
Vertical list, Each row displays:
- Report name (derived from filename)
- Optional metadata (e.g. company, year if parseable)
- Action button labeled: “Analyze” on the right of the row


### Report Multi-Select Modal Layout
- Button at top of modal which proceeds to chat session with selected files
- Make use of the `st_file_browser` component
- This feature is found here: https://github.com/pragmatic-streamlit/streamlit-file-browser
- Browsing The database should allow for a nested structure, specifically root/company/reports.



## Uploading Reports
- Clicking **“Add Your Own Report”**:
  - Opens a file upload modal

### Report Self-Upload Modal Layout
Form containing:
- File id name (what the file be reffered to as)
- Year published
- Company (If the comapny name exists in our database, while they type it in we offer autocomplete)
- File in .docx format
- Use `st.file_uploader()`

The info should be enough to make an entry in the *project_root*/data/index.json




---

## Transition: Report Selection → Chat Page

### Interaction
- Clicking **“Analyze”**:
- Closes the modal
- Navigates to a **new page**
- Selected report becomes the active context

This page transition is explicit (not hidden state).

---

## Main App Page: Chat Interface

### Purpose
This is the primary working view for the app.

All analysis happens here.

---

### Layout

Single-column layout (for now):

- Top: Report context
- Middle: Conversation history
- Bottom: Chat input
- Sidebar top for navigation, below it the current selected files are displayed

---

### Report Context (Top)
- Clearly display:
- Report name
- Optional helper text:
“You are analyzing this document.”

No “change report” option in MVP.

---

## Chat Interface

### Conversation Area
- Displays system and assistant messages
- Scrollable
- Empty state text before first question:
“Ask a question about this report to begin.”

---

### Input Area
- Fixed at bottom
- Single-line or multi-line text input
- Placeholder text:
> “Ask a question about this report…”

- Submit via Enter or button

---

## Scope Constraints (Very Important)

Explicitly **not included** in this version:
- Uploading documents
- Browsing databases
- Searching reports
- Multiple report selection
- Report switching
- Memory across reports
- User accounts
- Settings
- Inline citations or highlighting
- Styling themes

---

## Core Flow Summary

**Landing Page  
→ Click “Start Analyzing”  
→ Modal opens (blurred background)  
→ Select report from `data/reports`  
→ Click “Analyze”  
→ Navigate to chat page**

That is the entire MVP flow.
