# Report Assistant – Simplified User Flow (MVP v1)

## Entry Point: Landing Page

### Purpose
Immediately communicate:
1. What the app does
2. That it works on *existing reports*
3. A single, clear next action

No exploration, no configuration, no accounts.

This is only the beginning, it does not outline all the features planned for the app.

---

## Hero Section

- Large headline (centered):
  **“Boost Your Financial Analysis”**

- Supporting text (1–2 lines):
  “Analyze annual reports from leading publicly traded companies using natural language.”

- Single primary CTA:
  **“Start Analyzing”**

No secondary buttons, no navigation links.

---

## Transition: Landing → Report Selection

### Interaction
- Clicking **“Start Analyzing”**:
  - Opens a centered modal (pop-up)
  - Background content is blurred and dimmed
  - Focus is fully on report selection

The landing page itself remains visually present but inactive.

No page reload.

---

## Report Selection Modal

### Data Source (Strict)
- Reports are loaded **only** from:
```
project_root/data/reports/
```

- No uploads
- No external browsing
- No search
- No sorting
- No filtering

This directory is assumed to contain a small, curated set of reports.

---

### Modal Layout

**Header**
- Title: “Select a report to analyze”
- Optional short subtitle: “Choose one document to continue”

**Content**
- Vertical list (table-style)
- One row per report file

Each row displays:
- Report name (derived from filename)
- Optional metadata (e.g. company, year if parseable)
- **Action button labeled: “Analyze”**

The “Analyze” button occupies the rightmost column  
(where a date or file size might normally appear).

---

### Selection Rules
- Only **one report** can be selected
- No multi-select
- Clicking “Analyze” immediately proceeds
- No confirmation step

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

No sidebars in MVP.

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
