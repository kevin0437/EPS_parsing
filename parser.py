import os
import glob
import re
import csv
from bs4 import BeautifulSoup
from typing import List
import pandas as pd

# -----------------------------------------------------------------------------
# 1) UTILITIES: number normalization
# -----------------------------------------------------------------------------
def normalize_number(raw: str,force = False) -> float:
    """Convert '(4.50)' → -4.5 and strip '$' if present."""
    val = raw.replace('$', '').replace(',', '').strip()
    if val.startswith('('):
        if val.endswith(')'):
            val = val[1:-1]
        else:
            val = val[1:]  # remove leading '('
        return -float(val)

    return -float(val) if force else float(val)

KEYWORDS = [
    "eps", "earnings per share", "earnings/share","per diluted share",
    "basic eps", "basic earnings per share",
    "diluted eps", "diluted earnings per share",
    "net eps", "net earnings per share",
    "gaap eps", "gaap earnings per share",
    "non-gaap eps", "non-gaap earnings per share",
    "adjusted eps", "adjusted earnings per share",
    "pro forma eps", "pro-forma eps", "pro forma earnings per share",
    "loss per share", "basic loss per share", "diluted loss per share",
    "earnings per ads", "earnings per adr",
    "per share","Net income per share","per common share",
    "net income per share","per common share",
    "net income per common share", "net income per share",
    "basic net income per share", "diluted net income per share",
    "total eps", "total earnings per share",
    "net income (loss) per share", "net income (loss) per common share",
]
START_HEADERS = [
    r'Earnings per share',
    r'Financial Highlights',
    r'Results of Operations',
    r'Net Income\s*\(Loss\)\s*per share',
    r'Exhibit\s+99\.1',
]
END_HEADERS = r'^(Item\s+\d+\.|Exhibit\s+\d+)'



def find_num_around_keyword(
    text: str,
    kw_start: int,
    kw_end: int,
    window_size: int = 20,
    side: str = "both"  # "both" or "right"
) -> tuple[str, int]:
    """
    Extract a snippet around the keyword span [kw_start:kw_end].
    
    If side=="both":
      • Look up to window_size chars on both sides, but trim at
        the nearest sentence-end (". ", "; ") or newline.
    If side=="right":
      • Only look window_size chars to the right, trimming at
        sentence-end or newline.
    
    Returns (snippet, left_index) where left_index is the snippet's
    start in the original text.
    """
    length = len(text)

    # Determine left boundary
    if side == "both":
        left_limit = max(0, kw_start - window_size)
        # trim back to last ". ", "; ", or "\n"
        left_slice = text[left_limit:kw_start]
        last_dot   = left_slice.rfind('. ')
        last_semi  = left_slice.rfind('; ')
        last_comma = left_slice.rfind(', ')
        last_nl    = left_slice.rfind('\n')
        last_sym   = max(last_dot, last_semi,last_comma)
        left = left_limit + last_sym + 1 if last_sym != -1 else left_limit
    else:  # side == "right"
        left = kw_start

    # Determine right boundary
    right_limit = min(length, kw_end + window_size)
    right_slice = text[kw_end:right_limit]
    
    next_dot   = right_slice.find('. ')
    next_semi  = right_slice.find('; ')
    next_comma = right_slice.find(', ')
    next_nl    = right_slice.find('\n')
    # gather all non-negative positions
    candidates = [pos for pos in (next_dot, next_semi,next_comma) if pos != -1]
    if candidates:
        cut_pos = min(candidates)
        right = kw_end + cut_pos + 1
    else:
        right = right_limit

    snippet = text[left:right]
    return snippet, left


def find_text_around_keyword(
    text: str,
    kw_start: int,
    kw_end: int,
    window_size: int = 20,
    side: str = "both"  # "both" or "right"
) -> tuple[str, int]:
    """
    Extract a snippet around the keyword span [kw_start:kw_end].
    
    If side=="both":
      • Look up to window_size chars on both sides, but trim at
        the nearest sentence-end (". ", "; ") or newline.
    If side=="right":
      • Only look window_size chars to the right, trimming at
        sentence-end or newline.
    
    Returns (snippet, left_index) where left_index is the snippet's
    start in the original text.
    """
    length = len(text)

    # Determine left boundary
    if side == "both":
        left_limit = max(0, kw_start - window_size)
        # trim back to last ". ", "; ", or "\n"
        left_slice = text[left_limit:kw_start]
        last_dot   = left_slice.rfind('. ')
        last_semi  = left_slice.rfind('; ')
        last_comma = left_slice.rfind(', ')
        last_nl    = left_slice.rfind('\n')
        last_sym   = max(last_dot, last_semi)
        left = left_limit + last_sym + 1 if last_sym != -1 else left_limit
    else:  # side == "right"
        left = kw_start

    # Determine right boundary
    right_limit = min(length, kw_end + window_size)
    right_slice = text[kw_end:right_limit]
    
    next_dot   = right_slice.find('. ')
    next_semi  = right_slice.find('; ')
    next_comma = right_slice.find(', ')
    next_nl    = right_slice.find('\n')
    # gather all non-negative positions
    candidates = [pos for pos in (next_dot, next_semi) if pos != -1]
    if candidates:
        cut_pos = min(candidates)
        right = kw_end + cut_pos + 1
    else:
        right = right_limit

    snippet_left = text[left:kw_start]
    snippet_right = text[kw_end:right]
    return snippet_left, snippet_right, left
  

def extract_from_text(html: str) -> float|None:
    
    body_text  = html_to_text(html)
    table_text = html_table_text(html)
    
    
    kw_pat = "|".join(re.escape(k) for k in KEYWORDS)
    KW_RE  = re.compile(rf'\b(?:{kw_pat})\b', re.IGNORECASE)
    NUM_RE = re.compile(r'\(?\$?\d+\.\d+\)?')
    candidates = []
    
    def process(content: str, mode: str):
        """
        mode == 'both'  -> look left & right
        mode == 'right' -> only right-of-keyword numbers
        """
        src = 1 if mode == 'both' else 0
        
        for m_kw in KW_RE.finditer(content):
            ks, ke = m_kw.span()
            kw_text = m_kw.group(0).lower()

            # get snippet around keyword
            snippet, left = find_num_around_keyword(
                content, ks, ke, window_size=260, side=mode
            )
            
            
            if mode == 'both':
                txt_left,txt_right,_ = find_text_around_keyword(
                    content, ks, ke, 70, side = "both"
                )
            else:
                txt_left,txt_right,_  = find_text_around_keyword(
                    content, ks, ke, 70, side = "both"
                )
                
            txt_around = txt_left + kw_text + txt_right
           
            # scan for numbers in that snippet
            for m_num in NUM_RE.finditer(snippet):
                ns, ne = m_num.span()
                abs_start = left + ns
                abs_end   = left + ne
                raw_num   = m_num.group(0)
                
                txt_left_num = find_text_around_keyword(
                    content, abs_start, abs_end, 150, side = "both"
                )[0]
                
                # if mode=='right', skip any number to the left
                if mode == 'right' and abs_start < ke:
                    continue
                
                # side flag for ranking (0 = right, 1 = left)
                if mode == 'both':
                    side_flag = 0 if abs_start >= ke else 1
                else:
                    side_flag = 0  # right-only mode
                # gap distance
                gap = abs((abs_start if side_flag==0 else abs_end) - (ke if side_flag==0 else ks))

                # GAAP vs non-GAAP
                meas = 1 if re.search(r'non[- ]?gaap|core|book|adjusted|pro[- ]?forma?',
                                     txt_left_num, re.IGNORECASE) else 0
                
                txt_low = txt_around.lower()

                if 'basic' in txt_low or ("eps" in txt_low and "diluted" not in txt_low):
                    # Rule 1: basic over diluted
                    typ = 0
                elif 'diluted' in txt_low:
                    typ = 1
                elif re.search(r'\b(net|total)\b', txt_low):
                    # Rule 3: if there are multiple EPS values, net/total prevails
                    typ = 2
                elif 'loss per share' in txt_low:
                    # Rule 5: if only loss per share exists, use it (and force negative)
                    typ = 3
                else:
                    # Everything else: plain (GAAP) EPS or non‐GAAP “adjusted” EPS
                    typ = 4

                # force negative if “loss” appears
                force_neg = bool(" loss " in txt_left_num.lower() or " loss" in txt_left_num.lower() or "loss " in txt_left_num.lower())

                # record candidate
                candidates.append(((meas, typ, src, ks, gap, side_flag), raw_num, force_neg))
                #print(f"Found candidate: {((meas, typ, src, gap, ks, side_flag), raw_num, force_neg)} at ({txt_around})")
                
    # process body with two‐sided search
    process(body_text, 'both')
    #process table with right‐only search
    process(table_text, 'right')

    if not candidates:
        return None

    # pick best candidate
    candidates.sort(key=lambda x: x[0])
    _, best_raw, do_force = candidates[0]
    return normalize_number(best_raw, force=do_force)

def html_table_text(html: str) -> str:
    """
    Extract all text content from every HTML <table>, row by row.
    • Each <tr> becomes one line.
    • Within a row, all <th> and <td> texts are joined by a space.
    • Different rows are separated by newlines; different tables by a blank line.
    """
    soup = BeautifulSoup(html, 'html.parser')
    lines = []

    for table in soup.find_all('table'):
        for tr in table.find_all('tr'):
            # collect text from all header and data cells
            cells = tr.find_all(['th', 'td'])
            texts = [cell.get_text(" ", strip=True) for cell in cells if cell.get_text(strip=True)]
            if texts:
                # join cell texts with a space
                lines.append(" ".join(texts))
        # blank line after each table
        if lines and lines[-1] != "":
            lines.append("")

    # remove any trailing blank line
    if lines and lines[-1] == "":
        lines.pop()

    return "\n".join(lines)

def html_to_text(html: str) -> str:
    """
    Convert an HTML string to plain text.
    - Removes <script> and <style> content.
    - Preserves meaningful whitespace and line breaks for block-level elements.
    """
    soup = BeautifulSoup(html, 'html.parser')

    # remove script and style elements
    for tag in soup(['script', 'style']):
        tag.decompose()

    # Replace block-level tags with newlines to preserve structure
    for block in soup.find_all(['p', 'div', 'br', 'li', 'tr', 'table',
                                'h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
        block.insert_before("\n")
        block.insert_after("\n")

    # Get text
    text = soup.get_text()

    # Collapse consecutive whitespace/newlines
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n\n", text)

    # Strip leading/trailing whitespace on each line
    lines = [line.strip() for line in text.splitlines()]
    # Remove empty lines
    lines = [line for line in lines if line]

    return "\n".join(lines)

def html_table_df(html: str) :
    soup = BeautifulSoup(html, 'html.parser')
    tables_data = []
    
    for table in soup.find_all('table'):
        
        if table is None:
            raise ValueError("No <table> found in HTML.")

        # 3. Remove any <td> or <th> with style="display:none"
        for cell in table.find_all(['td', 'th']):
            style = cell.get('style', '')
            if 'display:none' in style.replace(' ', ''):
                cell.decompose()

        # 4. Build a list of rows, expanding colspans
        all_rows = []
        for tr in table.find_all('tr'):
            row = []
            for cell in tr.find_all(['td', 'th']):
                text = cell.get_text(strip=True)
                colspan = int(cell.get('colspan', 1))
                # repeat the text for each colspan so DataFrame columns align
                row.extend([text] * colspan)
            if any(item != '' for item in row):  # skip completely empty rows
                all_rows.append(row)
        if len(all_rows) == 0:
            continue
        # 5. Pad rows to the same length
        max_cols = max(len(r) for r in all_rows)
        normalized = [r + [''] * (max_cols - len(r)) for r in all_rows]

        # 6. Create DataFrame
        df = pd.DataFrame(normalized)
        tables_data.append(df)
        
    result = []    
    for df in tables_data:
        df_processed = df.apply(stitch_row_tokens_df, axis=1)
        result.append(df_processed)
        
    return result




def stitch_row_tokens_df(row: pd.Series) -> pd.Series:
    """
    Given a pandas Series representing a DataFrame row of cell-strings,
    merge any isolated currency symbols or parens with their neighboring numbers,
    but keep the output series the same length by writing merged-from cells as blank.
    """
    # Convert to list of strings
    tokens = row.fillna('').astype(str).tolist()
    n = len(tokens)
    # Prepare output with blanks
    stitched = [''] * n
    tokens = [
        re.sub(r'\s+', '', tok) if tok.strip().startswith('$') else tok
        for tok in tokens
    ]
    i = 0

    while i < n:
        cell = tokens[i].strip()

        # Merge '$' with next
        if cell == '$' and i + 1 < n:
            stitched[i] = '$' + tokens[i+1].strip()
            # leave stitched[i+1] as blank
            i += 2
            continue

        # Merge '(' with next
        if cell == '(' and i + 1 < n:
            stitched[i] = '(' + tokens[i+1].strip()
            i += 2
            continue

        # Merge ')' into previous if possible
        if cell == ')' and i > 0:
            stitched[i-1] = (stitched[i-1] or tokens[i-1]).rstrip() + ')'
            # leave stitched[i] blank
            i += 1
            continue

        # Merge ')%' into previous (strip %)
        if cell == ')%' and i > 0:
            stitched[i-1] = (stitched[i-1] or tokens[i-1]).rstrip() + ')'
            i += 1
            continue

        # Merge '%' into previous
        if cell == '%' and i > 0:
            stitched[i-1] = (stitched[i-1] or tokens[i-1]).rstrip() + '%'
            i += 1
            continue

        # Otherwise copy as-is
        stitched[i] = tokens[i]
        i += 1

    # Return as Series with same index
    return pd.Series(stitched, index=row.index)



def is_eps_header_row(row_text: str, prev_text: str = "") -> bool:
    """
    Return True if row_text (or prev_text) looks like an EPS header,
    by checking any of these term‐pairs:
      • earnings … per share
      • income … per share
      • net income … per common share
      • attributable to … common shareholders
    """
    text = row_text.lower()
    prev = prev_text.lower()
    
    # list of (first_term, second_term) to look for
    pairs = [
        ("earnings",        "per share"),
        ("income",          "per share"),
        ("net income",      "per share"),
        ("net income",      "per common share"),
        ("loss",            "per share"),
        ("loss",            "per common share"),
        ("earnings",        "per common share"),
        ("earnings loss",   "per share"),
        ("attributable to", "common shareholders"),
    ]

    # check in this row or in the previous
    for first, second in pairs:
        if first in text and second in text:
            return True
        if first in prev and second in prev:
            return True

    return False

def find_basic_eps_latest_year_dfs(dfs: List[pd.DataFrame]):
    """
    Given a list of DataFrames (one per HTML table), collect all Basic EPS candidates
    and return the one with highest priority based on:
      1) Literal 'EPS' mention
      2) GAAP (plain EPS) over non-GAAP (adjusted/core)
      3) Basic > Diluted > Net/Total > Loss per share > others
      4) Header match in same row > previous row
      5) Earlier table index > later
      6) Row closer to header > further
    """
    # Patterns
    
    eps_literal_re = re.compile(r'\bEPS\b', re.IGNORECASE)
    non_gaap_re = re.compile(r'non[- ]?gaap|core|book|adjusted|pro[- ]?forma?', re.IGNORECASE)
    year_re = re.compile(r'\b(\d{4})\b')

    candidates = []  # list of (priority_tuple, eps_value)

    for table_idx, df in enumerate(dfs):
        
        # Find header row
        header_idx = None
        for i, row in df.iterrows():
            years = []
            for cell in row:
                
                if not isinstance(cell, str):
                    continue
                # find *all* four-digit years in the cell
                for ystr in year_re.findall(cell):
                    years.append(int(ystr))
            if len(years) >= 2:
                header_idx = i
                break
        if header_idx is None:
            continue
        
        
        
        # Determine latest-year columns
        header_row = df.iloc[header_idx].astype(str).tolist()
        year_cols = []
        for j, cell in enumerate(df.iloc[header_idx].astype(str)):
            m = year_re.search(cell)
            if m:
                year_cols.append((j, int(m.group(1))))
        if not year_cols:
            continue
        max_year = max(y for _, y in year_cols)
        
        max_year_cols = [j for j, y in year_cols if y == max_year]
        
        # Scan rows below header for 'Basic'
        for row_idx in range(header_idx + 1, len(df)):
            row_vals = df.iloc[row_idx].astype(str).tolist()
            
            row_text = " ".join(row_vals).lower()
            prev_text = ""
            if row_idx > 0:
                prev_vals = df.iloc[row_idx-1].astype(str).tolist()
                prev_text = " ".join(prev_vals).lower()
                
            
           
            # Skip share-count rows
            if 'shares ' in row_text or ('shares ' in prev_text and 'basic' not in row_text) or 'continuing operations' in row_text:
                continue
            
            if not ('basic' in row_text or row_vals[0].lower()=='eps'):
                continue
            
            
            # Determine header match context
            is_eps_literal = bool(eps_literal_re.search(row_text) or eps_literal_re.search(prev_text))
            
            is_other_header = is_eps_header_row(row_text, prev_text)

            
            if not (is_eps_literal or is_other_header):
                continue
            
           
            
            # Evaluate each candidate column for numeric EPS
            for col_idx in max_year_cols:
                if col_idx >= df.shape[1]:
                    continue
                raw = str(df.iat[row_idx, col_idx]).strip()
                
                if not re.match(r'^\$?\(?\d', raw):
                    continue

                # Normalize number
                eps = normalize_number(raw)
                
                # GAAP vs non-GAAP
                meas = 1 if non_gaap_re.search(row_text) else 0
                gaap_flag = 1 - meas  # GAAP (0) -> 1, non-GAAP (1) -> 0

                # Basic/Diluted/Net/Total/Loss per share ranking
                if 'basic' in row_text or ('eps' in row_text and 'basic' not in row_text):
                    typ = 0
                elif 'diluted' in row_text:
                    typ = 1
                elif re.search(r'\b(net|total)\b', row_text):
                    typ = 2
                elif 'loss per share' in row_text:
                    typ = 3
                else:
                    typ = 4

                # Header same vs prev
                #has_header_same = int(other_header_re.search(row_text) is not None or eps_literal_re.search(row_text) is not None)

                # Build priority tuple
                priority = (
                    int(is_eps_literal),  # literal 'EPS' highest
                    gaap_flag,            # GAAP over non-GAAP
                    -typ,                 # lower typ better (basic=0 best)
                    #has_header_same,      # header in same row better
                    -table_idx,           # earlier table better
                    -row_idx,             # closer to header better
                    col_idx               # rightmost column wins tie
                )
                candidates.append((priority, eps))

    if not candidates:
        return None
    print(candidates)
    # Pick highest-priority candidate
    candidates.sort(reverse=True, key=lambda x: x[0])
    return candidates[0][1]


def extract_eps(html: str) -> float|None:

    # 1) First try finding it in a true parsed table:
    v = find_basic_eps_latest_year_dfs(html_table_df(html))
    
    if v is not None:
        print("Found EPS in table structure "+f": {v}")
        return v 

    # 2) Fallback to your existing full-text + table-text regex pass
    return extract_from_text(html)
    
# -----------------------------------------------------------------------------
# 6) BATCH PROCESS & CSV OUTPUT
# -----------------------------------------------------------------------------
def main():
    input_dir = "Training_Filings"
    output_csv = "output.csv"
    rows = []

    for path in glob.glob(os.path.join(input_dir, "*.html")):
        html = open(path, encoding="utf-8").read()
        eps  = extract_eps(html)
        rows.append([os.path.basename(path), eps])
        print("\n")
        print(f"{os.path.basename(path)} → {eps}")

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "eps"])
        writer.writerows(rows)

    print(f"\nDone. Results in {output_csv}")

if __name__ == "__main__":
    main()
