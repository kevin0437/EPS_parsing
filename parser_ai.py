import os
import glob
import csv
import openai
from openai import OpenAI
import json
import os
import glob
import re
import csv
from bs4 import BeautifulSoup
from typing import List
import pandas as pd

client = OpenAI(api_key = "your_api_key_here")  # Replace with your OpenAI API key

# Ensure you have set your OpenAI API key in the environment:
# export OPENAI_API_KEY="your_api_key_here"
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

# Directory containing HTML filings
INPUT_DIR = "Training_Filings"
# Output CSV file
OUTPUT_CSV = "output_ai.csv"

# System prompt describing the assistant's role and the EPS rules
SYSTEM_PROMPT = (
    "You are an expert financial data extractor. "
    "Extract the single latest EPS value from the provided filing, following these rules:\n"
    "1. If both diluted EPS and basic EPS are present, prioritize basic EPS.\n"
    "2. If both adjusted (Non-GAAP) EPS and unadjusted (GAAP) EPS are provided, output the unadjusted (GAAP) EPS.\n"
    "3. If multiple EPS instances appear, output the net or total EPS.\n"
    "4. Parentheses indicate negative values (e.g. (4.5) => -4.5).\n"
    "5. If no EPS but a loss per share is present, output that negative value.\n"
    "Provide only the numeric EPS value (e.g. -0.51 or 1.23), no other text."
)

# Iterate over HTML files and query the ChatGPT API
rows = []
for path in glob.glob(os.path.join(INPUT_DIR, "*.html")):
    html = open(path, encoding="utf-8").read()   
    
    filename = os.path.basename(path)
    
    body_text  = html_to_text(html)
    table_text = html_table_text(html)
    text = body_text + "\n\n" + table_text
    # Construct the user prompt
    user_prompt = (
        "Here is the text content of an SEC filing or press release:\n" + text + "\n\n"
        "Please extract the EPS according to the rules."
        "Respond *only* with a JSON object containing exactly one key: EPS, with the value being the EPS number.\n"
        "example: {\"EPS\": \"1.23\"}\n"
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt}
            ],
            temperature=0.0,
            max_tokens=50
        )
        eps_value = response.choices[0].message.content.strip()
    except Exception as e:
        eps_value = None
        print(f"Error processing {filename}: {e}")

    
    def extract_json_via_regex(text: str) -> dict:
            m = re.search(r'\{[\s\S]*?\}', text)
            print(f"Extracting JSON from: {text}")
            if not m:
                raise ValueError("No JSON found")
            return json.loads(m.group())
    print(f"{filename} -> {eps_value}")
    data = extract_json_via_regex(eps_value)
    eps = data.get("EPS", 0)
    
    rows.append([filename, eps])

# Write out results
with open(OUTPUT_CSV, "w", newline='', encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["file", "eps"] )
    writer.writerows(rows)

print(f"Done. Results written to {OUTPUT_CSV}")
