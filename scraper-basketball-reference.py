### Most code from https://www.youtube.com/watch?v=o6Ih934hADU

# Import packages
import os
import time
import asyncio
import sys
from bs4 import BeautifulSoup # python3 -m pip install --user beautifulsoup4
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeout # python3 -m pip install --user playwright & python3 -m playwright install

# Fixes a unicode error I was getting during scrape_game()
sys.stdin.reconfigure(encoding='utf-8')
sys.stdout.reconfigure(encoding='utf-8')

# Constants
SEASONS = list(range(2016, 2025)) # Array of years from 2016-2024
DATA_DIR = 'data'
STANDINGS_DIR = os.path.join(DATA_DIR, 'standings')
SCORES_DIR = os.path.join(DATA_DIR, 'scores')

# Function to get the html of a given page
async def get_html(url, selector, sleep=5, retries=3):
    html = None
    # Sometimes website will ban you, need retry logic in case of that
    for i in range(1, retries+1):
        time.sleep(sleep * i)

        # Initializes playwright instance and gets html given a selector
        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch()
                page = await browser.new_page()
                await page.goto(url)
                print(await page.title())
                html = await page.inner_html(selector)
        except PlaywrightTimeout:
            print(f'Timeout error on {url}')
            continue
        else:
            break
    return html

# Function to get the urls to each month of a given season
async def scrape_season(season):
    url = f'https://www.basketball-reference.com/leagues/NBA_{season}_games.html'
    html = await get_html(url, "#content .filter")

    # Parse the html and extract all hrefs
    soup = BeautifulSoup(html, features='html.parser')
    links = soup.find_all('a')
    href = [l['href'] for l in links]
    standings_pages = [f'https://www.basketball-reference.com{l}' for l in href]
    
    # Iterate through each standings page and save the html
    for url in standings_pages:
        save_path = os.path.join(STANDINGS_DIR, url.split('/')[-1])
        if os.path.exists(save_path):
            continue

        html = await get_html(url, '#all_schedule')
        with open(save_path, 'w+', encoding='utf-8', errors='replace') as f:
            f.write(html)

async def scrape_game(standings_file):
    with open(standings_file, 'r', encoding='utf-8', errors='replace') as f:
        html = f.read()

    soup = BeautifulSoup(html, features='html.parser')
    links = soup.find_all('a')
    hrefs = [l.get('href') for l in links]
    box_scores = [l for l in hrefs if l and 'boxscore' in l and '.html' in l]
    box_scores = [f'https://www.basketball-reference.com{l}' for l in box_scores]

    for url in box_scores:
        save_path = os.path.join(SCORES_DIR, url.split('/')[-1])
        if os.path.exists(save_path):
            continue

        html = await get_html(url, '#content')
        if not html:
            continue
        with open(save_path, 'w+', encoding='utf-8', errors='replace') as f:
            f.write(html)

async def main():
    # Create directories if they don't exist
    for directory in [DATA_DIR, SCORES_DIR, STANDINGS_DIR]:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created: {directory}")
        else:
            print(f"Already exists: {directory}")

    # Scrape the seasons html from seasons we don't have data from (First-run Time Estimate: 1 minute)
    for season in SEASONS:
        await scrape_season(season)
    
    # Scrape the scores html from months we don't have data from (First-run Time Estimate: 1.5 days)
    standings_files = os.listdir(STANDINGS_DIR)
    standings_files = [s for s in standings_files if '.html' in s]
    for f in standings_files:
        filepath = os.path.join(STANDINGS_DIR, f)
        await scrape_game(filepath)

if __name__ == '__main__':
    asyncio.run(main())  # Properly run async function