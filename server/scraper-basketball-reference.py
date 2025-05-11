# Import packages
import os
import time
import asyncio
import sys
from datetime import datetime
from bs4 import BeautifulSoup                      # python3 -m pip install --user beautifulsoup4
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeout  
                                                  # python3 -m pip install --user playwright & python3 -m playwright install

# Fixes a unicode error I was getting during scrape_game()
sys.stdin.reconfigure(encoding='utf-8')
sys.stdout.reconfigure(encoding='utf-8')

# Constants
SEASONS = list(range(2016, 2026))  # Array of years from 2016–2025
DATA_DIR = 'data'
STANDINGS_DIR = os.path.join(DATA_DIR, 'standings')
SCORES_DIR = os.path.join(DATA_DIR, 'scores')

async def get_html(url, selector, sleep=5, retries=3):
    """
    Load `url` in Playwright, wait for `selector`, return inner_html.
    Retries on Timeout up to `retries` times, sleeping between attempts.
    """
    html = None
    for i in range(1, retries + 1):
        # back‑off sleep between retries
        time.sleep(sleep * i)

        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch()
                page = await browser.new_page()
                await page.goto(url)
                print(f"Loaded: {await page.title()}")
                html = await page.inner_html(selector)
        except PlaywrightTimeout:
            print(f"Timeout fetching {url}, retry {i}/{retries}")
            continue
        else:
            break
    return html

async def scrape_season(season):
    """
    Historic scrape: for given season year, fetch each month overview
    and save to STANDINGS_DIR.
    """
    url = f'https://www.basketball-reference.com/leagues/NBA_{season}_games.html'
    html = await get_html(url, "#content .filter")
    if not html:
        return

    soup = BeautifulSoup(html, 'html.parser')
    links = soup.find_all('a')
    hrefs = [l['href'] for l in links]
    standings_pages = [f'https://www.basketball-reference.com{l}' for l in hrefs]

    for url in standings_pages:
        save_path = os.path.join(STANDINGS_DIR, url.split('/')[-1])
        if os.path.exists(save_path):
            print(f"Skip existing: {save_path}")
            continue

        page_html = await get_html(url, '#all_schedule')
        if page_html:
            with open(save_path, 'w+', encoding='utf-8', errors='replace') as f:
                f.write(page_html)
                print(f"Saved season page {save_path}")

async def scrape_season_month(season: int, month: str):
    """
    Fetch just the NBA_{season}_games-{month}.html and its box‑scores.
    """
    url = f'https://www.basketball-reference.com/leagues/NBA_{season}_games-{month}.html'
    print(f"→ refreshing standings for {season} {month.title()}")
    html = await get_html(url, "#all_schedule")
    if not html:
        print("  ! no html, skipping")
        return

    fname = f'NBA_{season}_games-{month}.html'
    save_path = os.path.join(STANDINGS_DIR, fname)
    with open(save_path, 'w+', encoding='utf-8', errors='replace') as f:
        f.write(html)
        print(f"  ✓ saved {save_path}")

    # now re‑scrape all box‑scores linked from that page
    await scrape_game(save_path)

async def scrape_game(standings_file):
    """
    Given a saved month-overview HTML file, find all boxscore links
    and save each into SCORES_DIR.
    """
    with open(standings_file, 'r', encoding='utf-8', errors='replace') as f:
        html = f.read()

    soup = BeautifulSoup(html, 'html.parser')
    # only <a> tags with an href
    links = soup.find_all('a', href=True)
    hrefs = [a['href'] for a in links]
    # group the or-test so h is never None when you do 'in'
    box_scores = [h for h in hrefs if ('boxscores' in h or 'boxscore' in h) and h.endswith('.html')]

    for h in box_scores:
        url = f'https://www.basketball-reference.com{h}'
        fname = url.split('/')[-1]
        save_path = os.path.join(SCORES_DIR, fname)
        if os.path.exists(save_path):
            continue
        page_html = await get_html(url, '#content')
        if not page_html:
            continue
        with open(save_path, 'w+', encoding='utf-8', errors='replace') as f:
            f.write(page_html)
            print(f"Saved box score {fname}")

async def main():
    # Create directories if they don't exist
    for directory in (DATA_DIR, STANDINGS_DIR, SCORES_DIR):
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created: {directory}")
        else:
            print(f"Already exists: {directory}")

    # Only refresh the latest season + current month
    latest_season = SEASONS[-1]
    current_month = datetime.now().strftime('%B').lower()  # e.g. "may"
    await scrape_season_month(latest_season, current_month)

    # If you ever want to backfill all seasons, uncomment:
    # for season in SEASONS[:-1]:
    #      await scrape_season(season)

if __name__ == '__main__':
    asyncio.run(main())  # Properly run async function
