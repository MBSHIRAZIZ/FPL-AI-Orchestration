
"""
FPL AI Orchestration — Improved prototype (v5).

Changes vs. v4:
  - Injury detection now honours recovery/return phrases
    ("returned from injury" no longer flags a player as injured).
  - Added alias "gross" -> "Groß" to map Whisper's ASCII output
    to the accented FPL web_name.

Install dependencies:
  pip install feedparser
"""

import os
import re
import difflib
import time

# --------------------------------------------------------------
# FFMPEG PATH (Windows / VS Code)
# --------------------------------------------------------------
FFMPEG_DIR = r"C:\ffmpeg\bin"   # <-- change to your path if different
if os.path.exists(os.path.join(FFMPEG_DIR, "ffmpeg.exe")):
    os.environ["PATH"] = FFMPEG_DIR + os.pathsep + os.environ["PATH"]
    print(f"✅ Manually added ffmpeg to PATH from: {FFMPEG_DIR}")
else:
    print(f"⚠️ Warning: ffmpeg not found at {FFMPEG_DIR}")

# Fix SSL certificate issues for Windows
import ssl
try:
    ssl._create_default_https_context = ssl._create_unverified_context
except Exception:
    pass

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import ollama
import whisper
import requests
import nltk
import feedparser
from nltk.sentiment import SentimentIntensityAnalyzer

nltk.download("vader_lexicon", quiet=True)

# ==============================================================
# CONFIG
# ==============================================================
AUDIO_FILE = r"C:\Users\mubas\Desktop\Final Year Project\FYP Mid term prototype\fplgw5crop.mp3"

WHISPER_MODEL = "base"
SUMMARY_MODEL = "llama3.1"

# ---- Fuzzy cutoffs (conservative) ----
FUZZY_CUTOFF_LONG  = 0.85
FUZZY_CUTOFF_MED   = 0.85
FUZZY_CUTOFF_SHORT = 0.95
INJURY_WINDOW_CHARS = 80

# ---- Reddit RSS config ----
REDDIT_SUBREDDIT   = "FantasyPL"
REDDIT_MAX_POSTS   = 15
REDDIT_TIME_FILTER = "month"     # hour | day | week | month | year | all

# ---- Manual alias map ----
# IMPORTANT: keys must already be normalised (lowercase, letters only).
PLAYER_ALIASES = {
    # Trossard variants
    "leotroisam":  "Trossard",
    "leotroissam": "Trossard",
    "troisam":     "Trossard",
    "troissam":    "Trossard",
    "trosard":     "Trossard",
    "trossad":     "Trossard",
    # Haaland variants
    "harlan":      "Haaland",
    "harlant":     "Haaland",
    "halland":     "Haaland",
    "halaand":     "Haaland",
    # Saka variants
    "sacca":       "Saka",
    "sakah":       "Saka",
    "sakar":       "Saka",
    # Bruno Fernandes variants
    "brunog":      "Bruno Fernandes",
    "brunoguimaraes": "Bruno Guimarães",
    # Groß (Brighton) — Whisper transcribes ß as "ss"
    "gross":       "Groß",
    "pascalgross": "Groß",
    "callumgross": "Groß",
}

# ---- Manager / non-player terms to ignore ----
NON_PLAYER_TERMS = {
    "guardiola", "guadalupe", "arteta", "klopp", "tenhag", "poch",
    "ancelotti", "mourinho", "conte", "howe", "emery", "postecoglou",
    "fpl", "premierleague", "facup", "championsleague",
}

# ==============================================================
# FPL DOMAIN KEYWORDS (for hybrid sentiment override)
# ==============================================================
NEGATIVE_PATTERNS = [
    r"\binjur(?:y|ed|ies)\b",
    r"\bdoubt(?:ful|s)?\b",
    r"\bflagged\b",
    r"\brotation risk\b",
    r"\bbenched\b",
    r"\bdropping in price\b",
    r"\bprice (?:drop|fall|plummet)\b",
    r"\bsell(?:ing)?\b",
    r"\btransfer(?:ring)?\s+\w+\s+out\b",
    r"\bavoid\b",
    r"\bout of form\b",
    r"\bpoor form\b",
    r"\bblank(?:ed|ing)?\b",
    r"\bsuspension\b",
    r"\bred card\b",
    r"\bnot (?:starting|nailed)\b",
    r"\bconcern(?:s|ed)?\b",
    r"\bunlikely\b",
    r"\bmiss(?:es|ing)\b",
]

POSITIVE_PATTERNS = [
    r"\bessential\b",
    r"\bnailed\b",
    r"\bin (?:great |good |excellent |impressive )?form\b",
    r"\bimpressive form\b",
    r"\bmust[- ]have\b",
    r"\bbuy\b",
    r"\btransfer(?:ring)?\s+(?:in|him|her)\b",
    r"\bdifferential\b",
    r"\bhaul(?:ed)?\b",
    r"\bfit\b",
    r"\brecommend(?:ed|ing)?\b",
    r"\bconsistent\b",
    r"\btop (?:pick|target)\b",
]

# ==============================================================
# PLAYER NAME INDEX + FUZZY MATCHING
# ==============================================================
def _normalise(text: str) -> str:
    """Lowercase and strip non-letters so \"O'Reilly\" -> \"oreilly\"."""
    return re.sub(r"[^a-z]", "", text.lower())


def _dynamic_cutoff(key: str) -> float:
    n = len(key)
    if n <= 4:
        return FUZZY_CUTOFF_SHORT
    if n <= 6:
        return FUZZY_CUTOFF_MED
    return FUZZY_CUTOFF_LONG


def build_player_index(players):
    """
    Build a lookup: normalised_name -> {'display': str, 'team_id': int}
    Uses only the primary web_name to avoid duplicate identity issues.
    """
    index = {}
    for p in players:
        first  = (p.get("first_name") or "").strip()
        second = (p.get("second_name") or "").strip()
        web    = (p.get("web_name") or "").strip()
        team_id = p.get("team")
        display = web or f"{first} {second}".strip()

        candidates = set()
        for cand in [web, second, f"{first} {second}", f"{second} {first}"]:
            cand = cand.strip()
            if len(cand) >= 4:
                candidates.add(cand)

        for cand in candidates:
            key = _normalise(cand)
            if len(key) < 3:
                continue
            # Prefer entries with longer display names (avoids collisions
            # like "Fernandes" mapping to a lesser-known Fernandes)
            if key in index:
                if len(display) < len(index[key]["display"]):
                    continue
            index[key] = {"display": display, "team_id": team_id}
    return index


def _bucket_index(player_index):
    """Group index keys by first letter for fast lookup."""
    buckets = {}
    for key, meta in player_index.items():
        buckets.setdefault(key[:1], []).append((key, meta))
    return buckets


def _lookup_player(key, player_index, buckets):
    """Exact match first, then alias map, then fuzzy match."""
    for term in NON_PLAYER_TERMS:
        if term in key:
            return None

    # 1. Alias map
    alias = PLAYER_ALIASES.get(key)
    if alias:
        alias_key = _normalise(alias)
        if alias_key in player_index:
            return player_index[alias_key]

    # 2. Exact match
    if key in player_index:
        return player_index[key]

    # 3. Fuzzy match within first-letter bucket
    bucket = buckets.get(key[:1], [])
    if not bucket:
        return None
    cutoff = _dynamic_cutoff(key)
    candidates = [k for k, _ in bucket if abs(len(k) - len(key)) <= 3]
    if not candidates:
        return None
    matches = difflib.get_close_matches(key, candidates, n=1, cutoff=cutoff)
    if matches:
        for k, meta in bucket:
            if k == matches[0]:
                return meta
    return None


def correct_player_names(text, player_index, buckets):
    """Replace mis-transcribed player names with the canonical FPL name."""
    tokens = list(re.finditer(r"[A-Za-z']+", text))
    n = len(tokens)
    if n == 0:
        return text, set()

    recognised = set()
    replacements = []

    i = 0
    while i < n:
        matched = False
        for size in (3, 2, 1):
            j = i + size - 1
            if j >= n:
                continue
            start = tokens[i].start()
            end   = tokens[j].end()
            gram  = text[start:end]
            key   = _normalise(gram)
            if len(key) < 4:
                continue
            match = _lookup_player(key, player_index, buckets)
            if match:
                replacements.append((start, end, match["display"]))
                recognised.add(match["display"])
                i += size
                matched = True
                break
        if not matched:
            i += 1

    corrected = text
    for start, end, repl in sorted(replacements, key=lambda r: r[0], reverse=True):
        corrected = corrected[:start] + repl + corrected[end:]

    return corrected, recognised


def find_players_with_context(text, player_index, buckets, window=INJURY_WINDOW_CHARS):
    """Returns list of {'name', 'mention', 'context'}."""
    tokens = list(re.finditer(r"[A-Za-z']+", text))
    n = len(tokens)
    results = []
    i = 0
    while i < n:
        matched = False
        for size in (3, 2, 1):
            j = i + size - 1
            if j >= n:
                continue
            start = tokens[i].start()
            end   = tokens[j].end()
            gram  = text[start:end]
            key   = _normalise(gram)
            if len(key) < 4:
                continue
            match = _lookup_player(key, player_index, buckets)
            if match:
                ctx_start = max(0, start - window)
                ctx_end   = min(len(text), end + window)
                results.append({
                    "name": match["display"],
                    "mention": gram,
                    "context": text[ctx_start:ctx_end],
                })
                i += size
                matched = True
                break
        if not matched:
            i += 1
    return results


# ==============================================================
# HYBRID SENTIMENT (VADER + FPL keyword override)
# ==============================================================
_SIA = SentimentIntensityAnalyzer()

def hybrid_sentiment(text):
    scores = _SIA.polarity_scores(text)
    compound = scores["compound"]

    neg_hits = sum(len(re.findall(p, text, re.IGNORECASE)) for p in NEGATIVE_PATTERNS)
    pos_hits = sum(len(re.findall(p, text, re.IGNORECASE)) for p in POSITIVE_PATTERNS)

    override = False
    final_score = compound
    if neg_hits > pos_hits:
        override = True
        final_score = min(compound - 0.4, -0.2)
    elif pos_hits > neg_hits:
        override = True
        final_score = max(compound + 0.4, 0.2)

    if final_score >= 0.05:
        label = "Positive"
    elif final_score <= -0.05:
        label = "Negative"
    else:
        label = "Neutral"

    return {
        "vader_compound": compound,
        "negative_hits": neg_hits,
        "positive_hits": pos_hits,
        "override_applied": override,
        "label": label,
        "score": round(final_score, 2),
    }


# ==============================================================
# MODEL 2: REDDIT RSS SENTIMENT
# ==============================================================
def get_reddit_sentiment_rss(player_name, max_posts=REDDIT_MAX_POSTS,
                              time_filter=REDDIT_TIME_FILTER):
    """
    Fetches recent posts from r/FantasyPL mentioning a player
    via Reddit's public RSS feed (no API key required).
    Returns dict: {'label', 'score', 'post_count', 'sample_titles', ...}
    """
    print(f"🔄 Fetching recent Reddit posts for '{player_name}'...")
    try:
        search_url = (
            f"https://www.reddit.com/r/{REDDIT_SUBREDDIT}/search.rss"
            f"?q={requests.utils.quote(player_name)}"
            f"&restrict_sr=1&sort=new&t={time_filter}&limit={max_posts}"
        )

        feed = feedparser.parse(search_url)

        if not feed.entries:
            print(f"   No recent Reddit posts found for '{player_name}'.")
            return {"label": "Neutral", "score": 0.0,
                    "post_count": 0, "sample_titles": []}

        all_text = []
        sample_titles = []
        for entry in feed.entries:
            title = getattr(entry, "title", "") or ""
            summary = getattr(entry, "summary", "") or ""
            clean_summary = re.sub(r"<[^>]+>", " ", summary)
            all_text.append(title)
            all_text.append(clean_summary)
            sample_titles.append(title.strip()[:120])

        combined_text = " ".join(all_text)
        result = hybrid_sentiment(combined_text)

        print(f"   Found {len(feed.entries)} posts about '{player_name}': "
              f"{result['label']} (score {result['score']})")
        if sample_titles:
            print(f"   Sample: {sample_titles[0][:90]}")

        return {
            "label": result["label"],
            "score": result["score"],
            "post_count": len(feed.entries),
            "sample_titles": sample_titles[:3],
            "vader_compound": result["vader_compound"],
            "override_applied": result["override_applied"],
        }

    except Exception as e:
        print(f"❌ Reddit RSS error for '{player_name}': {e}")
        return {"label": "Neutral", "score": 0.0,
                "post_count": 0, "sample_titles": []}


def get_sentiment_for_players(player_names):
    """
    Runs Reddit RSS sentiment for each player name.
    Returns (per_player_dict, overall_dict).
    """
    per_player = {}
    for name in player_names:
        per_player[name] = get_reddit_sentiment_rss(name)

    scores = [s["score"] for s in per_player.values() if s["post_count"] > 0]
    avg = sum(scores) / len(scores) if scores else 0.0
    overall_label = ("Positive" if avg >= 0.05
                     else "Negative" if avg <= -0.05
                     else "Neutral")
    return per_player, {"label": overall_label, "score": round(avg, 2)}


# ==============================================================
# MODEL 1: AUDIO -> TEXT -> SUMMARY
# ==============================================================
_WHISPER_CACHE = {}
def _get_whisper_model(name):
    if name not in _WHISPER_CACHE:
        _WHISPER_CACHE[name] = whisper.load_model(name)
    return _WHISPER_CACHE[name]


def transcribe_local_audio(audio_path, whisper_model=WHISPER_MODEL):
    audio_path = os.path.abspath(audio_path)
    print(f"🔄 Looking for file: {audio_path}")
    print(f"   File exists: {os.path.exists(audio_path)}")
    if not os.path.exists(audio_path):
        return None
    print("🔄 Loading Whisper model and transcribing...")
    try:
        model = _get_whisper_model(whisper_model)
        result = model.transcribe(audio_path)
        transcript = result["text"]
        print(f"✅ Transcription complete. Length: {len(transcript)} characters.")
        return transcript
    except Exception as e:
        print(f"❌ Transcription failed: {e}")
        return None


def summarize_transcript(transcript, player_index, buckets):
    if not transcript:
        return "No transcript available to summarize."

    clean_transcript, _ = correct_player_names(transcript, player_index, buckets)

    print(f"🔄 Summarizing {len(clean_transcript)} characters with "
          f"{SUMMARY_MODEL} (chunked)...")
    chunk_size = 2000
    chunks = [clean_transcript[i:i+chunk_size]
              for i in range(0, len(clean_transcript), chunk_size)]

    chunk_summaries = []
    for idx, chunk in enumerate(chunks):
        if len(chunk.strip()) < 50:
            continue
        try:
            response = ollama.chat(
                model=SUMMARY_MODEL,
                options={"temperature": 0.1},
                messages=[{
                    "role": "user",
                    "content": (
                        "Extract ONLY the key FPL injury news, player form updates, "
                        "and transfer recommendations from this text chunk. "
                        "Use the exact player names as written. Be specific. "
                        "Do NOT invent players or sentences not present in the text.\n\n"
                        f"{chunk}"
                    )
                }]
            )
            chunk_summaries.append(response["message"]["content"])
            print(f"   Summarised chunk {idx+1}/{len(chunks)}")
        except Exception as e:
            print(f"   Chunk {idx+1} failed: {e}")

    combined = "\n".join(chunk_summaries)
    print("🔄 Synthesising final summary...")
    try:
        final = ollama.chat(
            model=SUMMARY_MODEL,
            options={"temperature": 0.1},
            messages=[{
                "role": "user",
                "content": (
                    "Based on the extracted FPL notes below, produce exactly 3 "
                    "bullet points covering the most important injuries, form "
                    "updates, and transfer tips. Use the exact player names as "
                    "written. Do NOT invent players.\n\n"
                    f"{combined}"
                )
            }]
        )
        summary = final["message"]["content"]
    except Exception as e:
        summary = f"Ollama synthesis error: {e}"

    summary, _ = correct_player_names(summary, player_index, buckets)
    print("✅ Final summary generated (names corrected).")
    return summary


def analyze_audio(audio_path, player_index, buckets):
    transcript = transcribe_local_audio(audio_path)
    if not transcript:
        return "Failed to transcribe audio."
    return summarize_transcript(transcript, player_index, buckets)


# ==============================================================
# MODEL 3: FPL API + LINEAR REGRESSION
# ==============================================================
def fetch_fpl_data_direct():
    print("🔄 Fetching live FPL data from official API...")
    try:
        url = "https://fantasy.premierleague.com/api/bootstrap-static/"
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        data = r.json()
        players = data.get("elements", [])
        teams   = {t["id"]: t["name"] for t in data.get("teams", [])}
        print(f"✅ Fetched {len(players)} players.")
        return players, teams
    except Exception as e:
        print(f"❌ FPL API error: {e}")
        return None, None


def predict_fpl_points(players, teams):
    """
    Returns (top_df, full_df, mse, error_str).
    Predicts points per gameweek (not season totals).
    """
    if players is None:
        return None, None, None, "Failed to fetch FPL data."

    df = pd.DataFrame(players)

    required = ["form", "minutes", "points_per_game"]
    for col in required:
        if col not in df.columns:
            return None, None, None, f"Missing '{col}' column in FPL data."

    keep = ["web_name", "second_name", "team", "form",
            "minutes", "points_per_game", "total_points"]
    keep = [c for c in keep if c in df.columns]
    df = df[keep].copy()

    for c in ["form", "minutes", "points_per_game", "total_points"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["form", "minutes", "points_per_game"])
    df = df[df["minutes"] > 90].copy()

    if len(df) < 10:
        return None, None, None, f"Not enough data ({len(df)} rows)."

    X = df[["form", "minutes"]]
    y = df["points_per_game"]

    model = LinearRegression().fit(X, y)
    df["predicted_points"] = model.predict(X)

    df["name"] = df["web_name"].fillna(df["second_name"]).astype(str)
    df["team"] = df["team"].map(teams).fillna("Unknown")

    mse = mean_squared_error(y, df["predicted_points"])

    top = (df.sort_values("predicted_points", ascending=False)
             .head(20)
             [["name", "team", "predicted_points", "form",
               "minutes", "points_per_game", "total_points"]]
             .reset_index(drop=True))

    print(f"✅ Linear Regression trained. MSE: {mse:.3f}")
    print(f"   Top predicted player: {top.iloc[0]['name']} "
          f"({top.iloc[0]['team']}) — {top.iloc[0]['predicted_points']:.2f} pts/GW")
    return top, df, mse, None


# ==============================================================
# ORCHESTRATOR — per-player decision logic
# ==============================================================
def orchestrate(audio_summary, audio_players, sentiment_by_player,
                top_predictions, full_df, teams):
    if full_df is None or len(full_df) == 0:
        return []

    pred_lookup = {row["name"].lower(): (float(row["predicted_points"]),
                                          row["team"])
                   for _, row in full_df.iterrows()}
    top20 = set(top_predictions["name"].str.lower().tolist()) \
            if top_predictions is not None else set()

    mentioned = set()
    for p in audio_players:
        mentioned.add(p["name"])
    for n in sentiment_by_player:
        mentioned.add(n)

    decisions = []
    for name in sorted(mentioned):
        key = name.lower()
        info = pred_lookup.get(key)
        predicted = info[0] if info else None
        team = info[1] if info else "—"
        stat_strong = key in top20

        sent = sentiment_by_player.get(name,
                {"label": "Neutral", "score": 0.0, "post_count": 0})
        audio_ctx = next((p["context"] for p in audio_players
                          if p["name"] == name), "")

        # --- Injury detection with recovery/return negation ---
        recovery_in_audio = bool(re.search(
            r"\b(return(?:ed|s|ing)?\s+(?:from|to)|back\s+from|recovered|"
            r"available|passed fit|cleared to play|resumed training|"
            r"back in (?:the )?(?:team|squad|lineup|line-up))\b",
            audio_ctx, re.IGNORECASE
        ))
        raw_injury = bool(re.search(
            r"\b(injur(?:y|ed|ies)|doubt(?:ful)?|flagged|suspension|"
            r"out for|miss(?:es|ing)|not (?:fit|starting))\b",
            audio_ctx, re.IGNORECASE
        ))
        injury_in_audio = raw_injury and not recovery_in_audio

        recommend_in_audio = bool(re.search(
            r"\b(recommend(?:ed|ing)?|top pick|excellent form|"
            r"impressive form|great form|essential|must[- ]have|"
            r"lock[- ]in|differential)\b",
            audio_ctx, re.IGNORECASE
        ))

        # ---------- Decision rules ----------
        if sent["label"] == "Negative" and injury_in_audio:
            decision = "AVOID"
            reason = (f"Negative sentiment (score {sent['score']}) + "
                      f"injury/doubt in audio.")
        elif sent["label"] == "Negative" and sent["score"] <= -0.5:
            decision = "AVOID"
            reason = f"Strongly negative sentiment (score {sent['score']})."
        elif sent["label"] == "Negative":
            decision = "MONITOR"
            reason = f"Mildly negative sentiment (score {sent['score']})."
        elif injury_in_audio:
            decision = "MONITOR"
            reason = "Injury/doubt mentioned in audio."
        elif sent["label"] == "Positive" and stat_strong:
            decision = "TRANSFER IN"
            reason = (f"Positive sentiment (score {sent['score']}) + "
                      f"strong statistical prediction.")
        elif recommend_in_audio and stat_strong:
            decision = "CONSIDER"
            reason = "Recommended in audio + statistically strong."
        elif recommend_in_audio:
            decision = "CONSIDER"
            reason = "Recommended in audio."
        elif stat_strong:
            decision = "CONSIDER"
            reason = "Statistically strong; no negative signals."
        else:
            decision = "HOLD"
            reason = "No strong signals."

        decisions.append({
            "name": name, "team": team,
            "decision": decision, "reason": reason,
            "sentiment": sent["label"],
            "sentiment_score": sent["score"],
            "post_count": sent.get("post_count", 0),
            "injury_in_audio": injury_in_audio,
            "recovery_in_audio": recovery_in_audio,
            "recommend_in_audio": recommend_in_audio,
            "predicted_points": predicted,
            "stat_top20": stat_strong,
        })

    priority = {"AVOID": 0, "TRANSFER IN": 1, "CONSIDER": 2,
                "MONITOR": 3, "HOLD": 4}
    decisions.sort(key=lambda d: priority.get(d["decision"], 5))
    return decisions


# ==============================================================
# REPORT
# ==============================================================
def print_report(audio_summary, overall_sentiment, sentiment_by_player,
                 top_predictions, decisions, mse):
    print("\n" + "=" * 60)
    print("🧠 ORCHESTRATOR — FINAL RECOMMENDATION")
    print("=" * 60)

    print("\n🎵 [MODEL 1] Audio Summary (player names corrected)")
    print("-" * 60)
    print(audio_summary)

    print("\n🐦 [MODEL 2] Sentiment Analysis (Reddit r/FantasyPL)")
    print("-" * 60)
    print(f"   Overall: {overall_sentiment['label']} "
          f"(score {overall_sentiment['score']})")
    if sentiment_by_player:
        for n, s in sentiment_by_player.items():
            flag = " (override)" if s.get("override_applied") else ""
            print(f"   • {n}: {s['label']} (score {s['score']})"
                  f" — {s.get('post_count', 0)} posts{flag}")
    else:
        print("   No specific players detected in social text.")

    print("\n📊 [MODEL 3] Statistical Predictions")
    print("-" * 60)
    if mse is not None:
        print(f"   Model MSE: {mse:.3f}")
    if top_predictions is not None and len(top_predictions) > 0:
        print("   Top 5 by predicted points (per gameweek):")
        for i, row in top_predictions.head(5).iterrows():
            print(f"     {i+1}. {row['name']} ({row['team']}) — "
                  f"{row['predicted_points']:.2f} pts/GW")

    print("\n" + "=" * 60)
    print("🎯 PER-PLAYER DECISIONS")
    print("=" * 60)
    if not decisions:
        print("⚠️  No players were detected in the audio or social text.")
        return

    for d in decisions:
        emoji = {"AVOID": "🚫", "TRANSFER IN": "✅",
                 "CONSIDER": "💡", "MONITOR": "👀",
                 "HOLD": "⏸"}.get(d["decision"], "•")
        pts = ("—" if d["predicted_points"] is None
               else f"{d['predicted_points']:.2f}")
        print(f"\n{emoji} {d['name']} ({d['team']}) — {d['decision']}")
        print(f"    Reason: {d['reason']}")
        print(f"    Sentiment: {d['sentiment']} ({d['sentiment_score']})"
              f" [{d['post_count']} Reddit posts]"
              f"  | Injury mention: {d['injury_in_audio']}"
              f"  | Predicted pts: {pts}")
        if d.get("recovery_in_audio"):
            print(f"    (Note: recovery/return phrase detected — "
                  f"injury signal suppressed)")

    actionable = [d for d in decisions
                  if d["decision"] in ("AVOID", "TRANSFER IN", "CONSIDER")]
    print("\n" + "=" * 60)
    if actionable:
        print(f"✅ PIPELINE SUCCESSFUL — {len(actionable)} actionable "
              f"player recommendation(s) produced.")
    else:
        print("⚠️  Pipeline ran, but no actionable player recommendation "
              "was produced.")
    print("=" * 60)


# ==============================================================
# MAIN
# ==============================================================
def main():
    print("=" * 60)
    print("🎯 FPL AI PROTOTYPE — MULTI-MODAL ORCHESTRATION")
    print("   (Audio → Text → Summary | Reddit Sentiment | Tabular ML)")
    print("=" * 60)

    timings = {}

    # ---- 1. Load FPL data & build player index ----
    t0 = time.time()
    players, teams = fetch_fpl_data_direct()
    if players is None:
        print("❌ Could not load FPL data. Aborting.")
        return
    player_index = build_player_index(players)
    buckets = _bucket_index(player_index)
    timings["FPL API + index"] = time.time() - t0
    print(f"✅ Player index built ({len(player_index)} name keys).")

    # ---- 2. Model 1: Audio ----
    t0 = time.time()
    print(f"\n🎵 [MODEL 1] Analyzing audio: {AUDIO_FILE}")
    audio_summary = analyze_audio(AUDIO_FILE, player_index, buckets)
    timings["Audio pipeline"] = time.time() - t0

    audio_players = find_players_with_context(
        audio_summary, player_index, buckets
    )
    players_detected = sorted({p["name"] for p in audio_players})
    print(f"\n🔎 Players detected in audio summary: {players_detected}")

    # ---- 3. Model 2: Reddit RSS sentiment ----
    t0 = time.time()
    print("\n🐦 [MODEL 2] Analysing live Reddit sentiment...")
    sentiment_by_player, overall_sentiment = get_sentiment_for_players(
        players_detected
    )
    timings["Reddit sentiment"] = time.time() - t0

    # ---- 4. Model 3: Statistical prediction ----
    t0 = time.time()
    print("\n📊 [MODEL 3] Training Linear Regression on FPL data...")
    top_predictions, full_df, mse, err = predict_fpl_points(players, teams)
    if err:
        print(f"   ⚠️ {err}")
    timings["ML model"] = time.time() - t0

    # ---- 5. Orchestrate ----
    t0 = time.time()
    decisions = orchestrate(
        audio_summary, audio_players, sentiment_by_player,
        top_predictions, full_df, teams
    )
    timings["Orchestration"] = time.time() - t0

    # ---- 6. Report ----
    print_report(audio_summary, overall_sentiment, sentiment_by_player,
                 top_predictions, decisions, mse)

    # ---- 7. Timings ----
    print("\n⏱️  Stage timings:")
    for stage, secs in timings.items():
        print(f"   {stage:20s}: {secs:7.1f}s")
    print(f"   {'TOTAL':20s}: {sum(timings.values()):7.1f}s")


if __name__ == "__main__":
    main()