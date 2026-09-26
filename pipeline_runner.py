"""
Wrapper around prototype_fpl.py that runs the pipeline and returns
structured data suitable for rendering in a web page.
"""

import os
import time

import prototype_fpl as fpl


def _df_to_records(df):
    """Convert a pandas DataFrame to a list of plain dicts."""
    if df is None or len(df) == 0:
        return []
    return df.to_dict(orient="records")


def _safe_float(v):
    """Cast numpy float to Python float, or return None."""
    try:
        if v is None:
            return None
        return float(v)
    except (TypeError, ValueError):
        return None


def run_pipeline(audio_path=None):
    """
    Run the complete FPL pipeline and return a structured dict:

    {
        'audio_path': str,
        'audio': {'summary': str, 'detected_players': [...], 'contexts': [...]},
        'sentiment': {'per_player': {...}, 'overall': {...}},
        'ml': {'top_predictions': [...], 'mse': float or None},
        'orchestration': {'decisions': [...]},
        'timings': {'FPL API + index': float, ...},
        'errors': [...],
    }
    """
    if audio_path is None:
        audio_path = fpl.AUDIO_FILE

    result = {
        "audio_path": audio_path,
        "audio": {},
        "sentiment": {},
        "ml": {},
        "orchestration": {},
        "timings": {},
        "errors": [],
    }
    timings = {}

    # ---------- 1. FPL data + index ----------
    t0 = time.time()
    players, teams = fpl.fetch_fpl_data_direct()
    if players is None:
        result["errors"].append("Failed to fetch FPL data from the API.")
        return result

    player_index = fpl.build_player_index(players)
    buckets = fpl._bucket_index(player_index)
    timings["FPL API + index"] = round(time.time() - t0, 2)

    # ---------- 2. Audio ----------
    t0 = time.time()
    if not os.path.exists(audio_path):
        result["errors"].append("Audio file not found: {}".format(audio_path))
        audio_summary = ""
        audio_players = []
    else:
        audio_summary = fpl.analyze_audio(audio_path, player_index, buckets)
        audio_players = fpl.find_players_with_context(
            audio_summary, player_index, buckets
        )
    detected = sorted({p["name"] for p in audio_players})
    timings["Audio pipeline"] = round(time.time() - t0, 2)

    result["audio"] = {
        "summary": audio_summary or "(no audio processed)",
        "detected_players": detected,
        "contexts": audio_players,
    }

    # ---------- 3. Sentiment ----------
    t0 = time.time()
    sentiment_per_player, sentiment_overall = fpl.get_sentiment_for_players(
        detected
    )
    timings["Reddit sentiment"] = round(time.time() - t0, 2)

    result["sentiment"] = {
        "per_player": sentiment_per_player,
        "overall": sentiment_overall,
    }

    # ---------- 4. ML model ----------
    t0 = time.time()
    top_predictions, full_df, mse, err = fpl.predict_fpl_points(players, teams)
    if err:
        result["errors"].append("ML model: {}".format(err))
    timings["ML model"] = round(time.time() - t0, 2)

    # Convert DataFrame -> JSON-safe records
    top_records = _df_to_records(top_predictions)
    # Cast numpy floats to Python floats
    for rec in top_records:
        rec["predicted_points"] = _safe_float(rec.get("predicted_points"))
        rec["form"] = _safe_float(rec.get("form"))
        rec["points_per_game"] = _safe_float(rec.get("points_per_game"))

    result["ml"] = {
        "top_predictions": top_records,
        "mse": _safe_float(mse),
    }

    # ---------- 5. Orchestration ----------
    t0 = time.time()
    decisions = fpl.orchestrate(
        audio_summary, audio_players, sentiment_per_player,
        top_predictions, full_df, teams
    )
    timings["Orchestration"] = round(time.time() - t0, 2)

    # Cast numpy types in decisions
    for d in decisions:
        d["predicted_points"] = _safe_float(d.get("predicted_points"))
        d["sentiment_score"] = _safe_float(d.get("sentiment_score"))

    result["orchestration"] = {"decisions": decisions}

    result["timings"] = timings
    result["timings"]["TOTAL"] = round(sum(timings.values()), 2)

    return result