import os
import re
import math
import csv
import time
import random
import requests
from bs4 import BeautifulSoup
from datetime import datetime
import pandas as pd

# --- Configuration ---
SEARCH_QUERY = "laptop"  # change as needed
AMAZON_URL = f"https://www.amazon.in/s?k={SEARCH_QUERY}"
# File save directory: same directory as this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept-Language': 'en-US,en;q=0.9',
    'Accept-Encoding': 'gzip, deflate, br',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
    'Sec-Ch-Ua': '"Not_A Brand";v="8", "Chromium";v="120", "Google Chrome";v="120"',
    'Connection': 'keep-alive',
}

# --- Helper functions for robust extraction ---
def _round_half_up(value: float) -> int:
    return int(math.floor(value + 0.5))

def _extract_rating_from_text(raw: str):
    """Extract decimal rating and rounded integer star. Returns (rounded_str, raw_normalized) or (None,None)."""
    if not raw:
        return None, None
    raw = raw.strip()
    # common patterns like "4.3 out of 5 stars"
    m = re.search(r'(\d+(?:\.\d+)?)\s*(?:out of\s*5|of\s*5|/5|\u20135|\s*stars?)', raw, re.IGNORECASE)
    if m:
        try:
            val = float(m.group(1))
            rounded = _round_half_up(val)
            return str(rounded), f"{val} out of 5"
        except:
            pass
    # fallback: any standalone 1..5 number
    m2 = re.search(r'([1-5](?:\.\d+)?)', raw)
    if m2:
        try:
            val = float(m2.group(1))
            if 0.5 <= val <= 5.0:
                return str(_round_half_up(val)), f"{val} out of 5"
        except:
            pass
    return None, None

def _get_image_url(img_elem):
    """Return best available image URL handling lazy attributes."""
    if img_elem is None:
        return 'N/A'
    for attr in ('src', 'data-src', 'data-old-hires'):
        if img_elem.has_attr(attr) and img_elem.get(attr):
            return img_elem.get(attr)
    if img_elem.has_attr('data-a-dynamic-image'):
        val = img_elem.get('data-a-dynamic-image')
        m = re.search(r'"(https?://[^"]+)"', val)
        if m:
            return m.group(1)
    return img_elem.get('src') or 'N/A'

def extract_rating_from_card(product_card):
    """Robust rating extractor returning (rating_int_str_or_NA, raw_text_or_NA)."""
    selectors = [
        'div.a-icon-row span.a-icon-alt',
        'span.a-icon-alt',
        'span[data-hook="rating-out-of-text"]',
        'i[aria-label*="out of 5"]',
        'span[aria-label*="out of 5"]',
        'a[title*="out of 5"]',
        'span[title*="out of 5"]',
        'div[aria-label*="out of 5"]',
        'span.a-icon-star-small span.a-icon-alt'
    ]
    # Try selectors
    for sel in selectors:
        el = product_card.select_one(sel)
        if el:
            raw = el.get('title') or el.get('aria-label') or el.get_text(separator=' ', strip=True)
            rated, rawnorm = _extract_rating_from_text(raw)
            if rated:
                return rated, rawnorm
    # Search attributes on all children
    for el in product_card.select('*'):
        for attr in ('aria-label', 'title', 'alt', 'data-tooltip'):
            if el.has_attr(attr):
                val = el.get(attr)
                rated, rawnorm = _extract_rating_from_text(val)
                if rated:
                    return rated, rawnorm
    # Whole card text fallback
    text_blob = product_card.get_text(" ", strip=True)
    rated, rawnorm = _extract_rating_from_text(text_blob)
    if rated:
        return rated, rawnorm
    return 'N/A', 'N/A'

def get_product_details(product_card):
    """Return dictionary with product details for one product card."""
    data = {}
    # Ad / Organic
    if product_card.get('data-component-type') == 's-search-result':
        data['Ad / Organic Result'] = 'Organic'
    elif product_card.select_one('div.a-section.a-text-right span.a-badge-label-inner.a-text-ellipsis') or product_card.select_one('span.sponsored-label-text'):
        data['Ad / Organic Result'] = 'Ad'
    else:
        data['Ad / Organic Result'] = 'N/A'
    # Title
    title = None
    title_link = product_card.select_one('h2 a.a-link-normal')
    if title_link:
        title = title_link.get_text(strip=True)
    if not title:
        fallback_title = product_card.select_one('span.a-size-medium.a-color-base.a-text-normal') \
                         or product_card.select_one('span.a-size-base-plus.a-color-base.a-text-normal') \
                         or product_card.select_one('span.a-size-base.a-color-base')
        if fallback_title:
            title = fallback_title.get_text(strip=True)
    if not title:
        img_el = product_card.select_one('img.s-image')
        if img_el and img_el.get('alt'):
            title = img_el.get('alt').strip()
    data['Title'] = title or 'N/A'
    # Rating
    rated, raw = extract_rating_from_card(product_card)
    data['Rating'] = rated if rated else 'N/A'
    data['Rating_raw'] = raw if raw else 'N/A'
    # Price
    price_whole = product_card.select_one('span.a-price-whole')
    price_fraction = product_card.select_one('span.a-price-fraction')
    if price_whole and price_fraction:
        price_text = f"₹{price_whole.get_text(strip=True)}.{price_fraction.get_text(strip=True)}"
    else:
        price_offscreen = product_card.select_one('span.a-price span.a-offscreen')
        price_text = price_offscreen.get_text(strip=True) if price_offscreen else 'N/A'
    data['Price'] = price_text
    # Image URL
    img_elem = product_card.select_one('img.s-image')
    data['Image URL'] = _get_image_url(img_elem)
    return data

# --- Scraper main ---
def scrape_amazon(url):
    wait_time = random.uniform(2, 5)
    print(f"--- Sending request to: {url} (waiting {wait_time:.2f}s) ---")
    time.sleep(wait_time)
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
    except requests.exceptions.HTTPError:
        print(f"HTTP {resp.status_code} received. Amazon may be blocking the request.")
        return []
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return []
    soup = BeautifulSoup(resp.content, 'html.parser')
    product_cards = soup.find_all('div', {'data-component-type': 's-search-result'})
    if not product_cards:
        if soup.select_one('form[action="/errors/validateCaptcha"]'):
            print("CAPTCHA detected on page. Cannot scrape without solving a captcha or using a proxy.")
            return []
        # fallback selectors
        product_cards = soup.select('div.s-main-slot div.s-result-item')
        if not product_cards:
            print("No product cards found. Selector likely changed.")
            return []
    print(f"Found {len(product_cards)} product cards. Extracting...")
    out = []
    for card in product_cards:
        try:
            details = get_product_details(card)
            out.append(details)
        except Exception as e:
            print("Error parsing card:", e)
            continue
    return out

def save_to_csv(data, query):
    if not data:
        print("No data to save.")
        return None
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(BASE_DIR, f"amazon_{query}_data_{timestamp}.csv")
    try:
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False, encoding='utf-8', quoting=csv.QUOTE_ALL)
        print(f"✅ Saved CSV to: {filename}")
        return filename
    except Exception as e:
        print("Failed to save CSV:", e)
        return None

# --- Execute ---
if __name__ == "__main__":
    all_products = scrape_amazon(AMAZON_URL)
    if all_products:
        save_to_csv(all_products, SEARCH_QUERY)
    else:
        print("No products scraped.")

