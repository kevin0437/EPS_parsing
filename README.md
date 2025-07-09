## EPS Extraction Tool

This project processes HTML filings and extracts the Earnings Per Share (EPS) values using two complementary strategies: a structured table parser and a fallback full-text search. \
parser.py is the parser code. \
output.csv is the result \
parser_ai.py is the parser build with OpenAi API \
output_ai.csv is the result from the OpanAi API \
I used the result from OpenAi API to justify my parser's result when building the parser 

---

### 1. Installation & Setup

1. **Clone this repository** to your local machine.
2. **Install dependencies** (e.g., BeautifulSoup, pandas):
   ```bash
   pip install beautifulsoup4 pandas
   ```
3. **Prepare HTMLs**: place all `.html` files under a folder named `Training_Filings` in the project root.

---

### 2. Utilities: Number Normalization

- ``
  - Strips `$` and `,` from strings.
  - Converts parentheses `(4.50)` to negative values `-4.5`.
  - If `force=True`, always returns a negative number.

---

### 3. Full-Text & Snippet Extraction

#### 3.1 HTML → Plain Text

- ``
  - Removes `<script>` and `<style>` blocks.
  - Inserts newlines around block elements (`<p>`, `<div>`, `<table>`, etc.).
  - Collapses extra whitespace and cleans up blank lines.

#### 3.2 HTML → Table Text

- ``
  - Parses each `<table>` with BeautifulSoup.
  - Joins each row’s `<th>`/`<td>` content with spaces, one row per line.
  - Separates tables with blank lines.

#### 3.3 Snippet Windows

- `` and ``
  - Given a keyword span, extract a snippet of text around it (left, right, or both).
  - Trim at sentence boundaries (`. `, `; `, `, ` or newline).
  - Used to locate potential numeric candidates near EPS keywords.

---

### 4. Full-Text Regex-Based EPS Extraction

- ``

1. Convert HTML into two strings: `body_text` and `table_text`.
2. Compile a keyword regex (all EPS-related terms) and a number regex.
3. `` runs twice:
   - **Mode **`` on the full body: looks for numbers on either side of keywords.
   - **Mode **`` on table text: only numbers to the right of keywords.
4. For each keyword match:
   - Extract snippet and surrounding text.
   - Find all number matches in the snippet.
   - Compute ranking features:
     - ``: GAAP vs non‑GAAP.
     - ``: basic vs diluted vs net vs loss vs other.
     - ``: snippet source (body or table).
     - ``: distance from keyword.
     - ``: left vs right side.
   - Append `(features, raw_number, force_neg)` to candidate list.
5. Sort candidates by the tuple (lowest features tuple = highest priority).
6. Return `normalize_number(best_raw, force=force_neg)`.

---

### 5. Structured Table Parser

#### 5.1 Building DataFrames

- ``

  - Extract each HTML table into a pandas DataFrame.
  - Removes hidden cells (`display:none`).
  - Expands `colspan` by repeating text cells.
  - Pads rows for consistent column counts.
  - Applies `stitch_row_tokens_df` to merge split currency tokens.

- `` merges isolated `$`, `(`, `)`, `%` tokens with neighbors.

#### 5.2 Finding EPS in Tables

- ``

1. For each table DataFrame:
   - **Locate header row**: first row containing at least two 4-digit years.
   - **Identify columns** matching the latest year.
2. Scan rows below the header:
   - Skip rows about share counts or continuing operations.
   - Look for “basic” or literal “EPS” in the row or previous row.
   - Ensure the row context matches EPS-like headers.
3. For each candidate cell in the latest-year columns:
   - Validate numeric format.
   - Compute:
     - ``: GAAP vs non‑GAAP.
     - ``: basic/diluted/net/loss/other ranking.
     - Presence of literal “EPS”.
     - Table and row proximity penalties.
   - Build a priority tuple: literal EPS, GAAP, basic-first, table order, row closeness, column index.
4. Pick the highest-priority EPS value.

---

### 6. Unified Extraction Logic

- ``
  1. Try the structured table parser (`find_basic_eps_latest_year_dfs`).
  2. If that returns a value, use it.
  3. Otherwise, fall back to the full-text regex approach (`extract_from_text`).

---

### 7. Batch Processing & CSV Output

- ``:
  1. Scans all `*.html` files under `Training_Filings/`.
  2. Calls `extract_eps(html)` on each file.
  3. Writes results to `output.csv` with columns `[file, eps]`.

Run:

```bash
python your_script.py
```




