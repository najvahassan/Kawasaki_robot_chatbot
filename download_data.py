

import os
import csv
import requests
from pathlib import Path
from bs4 import BeautifulSoup

# Define CSV and download directory
csv_path =r"d:\Projects\ai_projects\Kawasaki_chatbot\kawasaki_robot_data.csv"
download_dir = "kawasaki_manuals"

#  Define sanitize() FIRST

def sanitize(name):
    """Return a safe filename."""
    return "".join(c for c in name if c.isalnum() or c in ['_', '-'])

# -------------------------------
# Define find_pdf() SECOND
# -------------------------------
def find_pdf(html, base_url):
    soup = BeautifulSoup(html, "html.parser")
    for a in soup.find_all("a", href=True):
        if a["href"].lower().endswith(".pdf"):
            return requests.compat.urljoin(base_url, a["href"])
    return None

# Create directory
os.makedirs(download_dir, exist_ok=True)

# -------------------------------
# MAIN PDF DOWNLOAD LOOP
# -------------------------------
with open(csv_path, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        url = row["Manual / PDF URL"]
        model = sanitize(row["Robot Model"].upper())

        print(f"Fetching: {url}")
        r = requests.get(url, timeout=20)

        # If direct PDF
        if "application/pdf" in r.headers.get("Content-Type", ""):
            out = Path(download_dir) / f"{model}.pdf"
            out.write_bytes(r.content)
            print(f"Saved PDF: {out}")
            continue

        # If HTML page
        pdf_url = find_pdf(r.text, url)
        if pdf_url:
            print(f"Found PDF link: {pdf_url}")
            r2 = requests.get(pdf_url)
            out = Path(download_dir) / f"{model}.pdf"
            out.write_bytes(r2.content)
            print(f"Saved extracted PDF: {out}")
            continue

        # Save HTML snapshot
        html_out = Path(download_dir) / f"{model}.html"
        html_out.write_text(r.text, encoding="utf-8", errors="replace")
        print(f"Saved HTML snapshot: {html_out}")
      