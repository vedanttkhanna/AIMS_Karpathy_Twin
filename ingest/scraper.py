import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import requests
import time
from pathlib import Path
from youtube_transcript_api import YouTubeTranscriptApi
from bs4 import BeautifulSoup
from config import (
    RAW_DATA_DIR,
    KARPATHY_BLOG_URLS,
    GITHUB_READMES,
    YOUTUBE_VIDEO_IDS,
)


def save(filename: str, content: str):
    path = Path(RAW_DATA_DIR) / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"  saved → {path}")


def scrape_blogs():
    print("\n[1/3] Scraping blog posts...")
    for url in KARPATHY_BLOG_URLS:
        try:
            resp = requests.get(url, timeout=10)
            soup = BeautifulSoup(resp.text, "html.parser")

            # remove nav, footer, scripts
            for tag in soup(["script", "style", "nav", "footer", "header"]):
                tag.decompose()

            text = soup.get_text(separator="\n", strip=True)
            slug = url.rstrip("/").split("/")[-1] or "index"
            save(f"blog_{slug}.txt", f"SOURCE: {url}\n\n{text}")
            time.sleep(0.5)
        except Exception as e:
            print(f"  failed {url}: {e}")


def scrape_readmes():
    print("\n[2/3] Scraping GitHub READMEs...")
    for url in GITHUB_READMES:
        try:
            resp = requests.get(url, timeout=10)
            # https://raw.githubusercontent.com/karpathy/REPONAME/master/README.md
            # index:  0      1                   2        3        4       5
            project = url.split("/")[4]
            save(f"readme_{project}.md", f"SOURCE: {url}\n\n{resp.text}")
            print(f"  saved readme for {project}")
            time.sleep(0.5)
        except Exception as e:
            print(f"  failed {url}: {e}")


def scrape_youtube():
    print("\n[3/3] Scraping YouTube transcripts...")
    for video_id in YOUTUBE_VIDEO_IDS:
        try:
            ytt = YouTubeTranscriptApi()
            fetched = ytt.fetch(video_id)
            text = "\n".join(
                f"[{int(t.start)}s] {t.text}" for t in fetched
            )
            url = f"https://www.youtube.com/watch?v={video_id}"
            save(f"youtube_{video_id}.txt", f"SOURCE: {url}\n\n{text}")
            print(f"  scraped video {video_id}")
            time.sleep(0.5)
        except Exception as e:
            print(f"  failed {video_id}: {e}")


def run_all():
    print("Starting data collection for Karpathy Twin...")
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    scrape_blogs()
    scrape_readmes()
    scrape_youtube()
    files = list(Path(RAW_DATA_DIR).glob("*"))
    print(f"\nDone. {len(files)} files saved to {RAW_DATA_DIR}/")


if __name__ == "__main__":
    run_all()