"""
Unit and integration tests for the FPL AI Orchestration pipeline.

Run with:
    pytest tests.py -v
    pytest tests.py -v -m "not slow"    # skip network/ASR-heavy tests
"""

import os
import pandas as pd
import pytest

# Import the module under test — adjust the filename if needed
import prototype_fpl as fpl


# ==============================================================
# 1. Text normalisation
# ==============================================================
class TestNormalise:

    def test_lowercase(self):
        assert fpl._normalise("SAKA") == "saka"

    def test_strips_punctuation(self):
        assert fpl._normalise("O'Reilly") == "oreilly"

    def test_strips_accents_keeps_letters(self):
        # Non-ASCII characters are removed (ß is not [a-z])
        assert fpl._normalise("Groß") == "gro"

    def test_handles_spaces(self):
        assert fpl._normalise("Bruno Fernandes") == "brunofernandes"

    def test_empty_string(self):
        assert fpl._normalise("") == ""


# ==============================================================
# 2. Player alias resolution
# ==============================================================
class TestAliasResolution:

    def test_haaland_alias(self):
        assert "harlan" in fpl.PLAYER_ALIASES
        assert fpl.PLAYER_ALIASES["harlan"] == "Haaland"

    def test_trossard_alias(self):
        assert fpl.PLAYER_ALIASES["leotroisam"] == "Trossard"

    def test_gross_alias(self):
        assert fpl.PLAYER_ALIASES["gross"] == "Groß"

    def test_alias_keys_are_normalised(self):
        """Every alias key must itself be a valid normalised string."""
        for key in fpl.PLAYER_ALIASES:
            assert key == fpl._normalise(key), (
                f"Alias key '{key}' is not normalised — "
                f"should be '{fpl._normalise(key)}'"
            )


# ==============================================================
# 3. Sentiment override
# ==============================================================
class TestSentimentOverride:

    def test_clear_negative(self):
        result = fpl.hybrid_sentiment(
            "Saka is injured and definitely dropping in price."
        )
        assert result["label"] == "Negative"
        assert result["score"] < 0

    def test_clear_positive(self):
        result = fpl.hybrid_sentiment(
            "Salah is essential and in excellent form."
        )
        assert result["label"] == "Positive"
        assert result["score"] > 0

    def test_neutral(self):
        result = fpl.hybrid_sentiment(
            "I am thinking about my team for next week."
        )
        assert result["label"] == "Neutral"

    def test_override_fires_on_negative_keyword(self):
        result = fpl.hybrid_sentiment(
            "Saka is flagged and might be a rotation risk."
        )
        assert result["override_applied"] is True
        assert result["label"] == "Negative"

    def test_override_fires_on_positive_keyword(self):
        result = fpl.hybrid_sentiment(
            "Palmer is nailed and a must-have in midfield."
        )
        assert result["override_applied"] is True
        assert result["label"] == "Positive"


# ==============================================================
# 4. Decision rules (orchestrator)
# ==============================================================
class TestDecisionRules:
    """
    Build minimal inputs for orchestrate() directly, rather than
    invoking predict_fpl_points() which requires >=10 players.
    """

    def _make_decision_input(self, sent_label="Neutral", sent_score=0.0,
                             injury=False, recommend=False, stat_strong=True):
        top = pd.DataFrame([{
            "name": "TestPlayer",
            "team": "Test FC",
            "predicted_points": 6.0,
            "form": 5.0,
            "minutes": 900,
            "points_per_game": 6.0,
            "total_points": 60,
        }])
        full = top.copy()
        teams = {1: "Test FC"}

        context_parts = []
        if injury:
            context_parts.append("TestPlayer is injured.")
        if recommend:
            context_parts.append("TestPlayer is recommended.")
        context_parts.append("TestPlayer featured in the discussion.")

        audio_players = [{
            "name": "TestPlayer",
            "mention": "TestPlayer",
            "context": " ".join(context_parts),
        }]
        sentiment = {"TestPlayer": {
            "label": sent_label,
            "score": sent_score,
            "post_count": 5,
        }}
        return audio_players, sentiment, top, full, teams

    def test_negative_plus_injury_avoids(self):
        ap, sent, top, full, teams = self._make_decision_input(
            sent_label="Negative", sent_score=-0.6, injury=True
        )
        decisions = fpl.orchestrate("summary", ap, sent, top, full, teams)
        assert len(decisions) > 0, "orchestrate returned no decisions"
        assert any(d["decision"] == "AVOID" for d in decisions), (
            f"Expected AVOID, got {[d['decision'] for d in decisions]}"
        )

    def test_positive_plus_stat_transfer_in(self):
        ap, sent, top, full, teams = self._make_decision_input(
            sent_label="Positive", sent_score=0.6, stat_strong=True
        )
        decisions = fpl.orchestrate("summary", ap, sent, top, full, teams)
        assert len(decisions) > 0, "orchestrate returned no decisions"
        assert any(d["decision"] == "TRANSFER IN" for d in decisions), (
            f"Expected TRANSFER IN, got {[d['decision'] for d in decisions]}"
        )

    def test_recovery_suppresses_injury(self):
        ap, sent, top, full, teams = self._make_decision_input(
            injury=False, recommend=True, stat_strong=True
        )
        # Inject recovery phrase
        ap[0]["context"] = "TestPlayer returned from injury and scored."
        decisions = fpl.orchestrate("summary", ap, sent, top, full, teams)
        assert len(decisions) > 0, "orchestrate returned no decisions"
        for d in decisions:
            assert d["injury_in_audio"] is False, (
                f"Injury flag should be suppressed, got {d}"
            )

    def test_negative_only_monitors(self):
        """Strongly negative sentiment without injury -> MONITOR (not AVOID)."""
        ap, sent, top, full, teams = self._make_decision_input(
            sent_label="Negative", sent_score=-0.3,
            injury=False, recommend=False, stat_strong=False
        )
        decisions = fpl.orchestrate("summary", ap, sent, top, full, teams)
        assert len(decisions) > 0, "orchestrate returned no decisions"
        assert any(d["decision"] == "MONITOR" for d in decisions), (
            f"Expected MONITOR, got {[d['decision'] for d in decisions]}"
        )

    def test_no_signals_holds(self):
        """No signals at all -> HOLD."""
        ap, sent, top, full, teams = self._make_decision_input(
            sent_label="Neutral", sent_score=0.0,
            injury=False, recommend=False, stat_strong=False
        )
        # Empty top DataFrame so stat_strong is False
        top_empty = top.iloc[0:0]
        decisions = fpl.orchestrate("summary", ap, sent, top_empty, full, teams)
        assert len(decisions) > 0, "orchestrate returned no decisions"
        assert any(d["decision"] == "HOLD" for d in decisions), (
            f"Expected HOLD, got {[d['decision'] for d in decisions]}"
        )

    def test_empty_full_df_returns_empty(self):
        """Guard: orchestrator with no full_df should return []."""
        ap, sent, top, full, teams = self._make_decision_input()
        decisions = fpl.orchestrate("summary", ap, sent, top, None, teams)
        assert decisions == []


# ==============================================================
# 5. FPL data integration
# ==============================================================
class TestFPLData:

    def test_fetch_returns_players(self):
        players, teams = fpl.fetch_fpl_data_direct()
        assert players is not None
        assert len(players) > 100
        assert teams is not None
        assert len(teams) > 10

    def test_predictions_are_realistic(self):
        players, teams = fpl.fetch_fpl_data_direct()
        top, full, mse, err = fpl.predict_fpl_points(players, teams)
        assert err is None, f"predict_fpl_points returned error: {err}"
        assert 0 < mse < 5, f"Per-GW MSE out of expected range: {mse}"
        # Top predicted should be between 5 and 15 pts/GW
        assert 5 < top.iloc[0]["predicted_points"] < 15, (
            f"Top prediction unrealistic: {top.iloc[0]['predicted_points']}"
        )

    def test_player_index_has_known_players(self):
        players, teams = fpl.fetch_fpl_data_direct()
        index = fpl.build_player_index(players)
        # We don't know which season this is, so just check the index is populated
        assert len(index) > 500


# ==============================================================
# 6. Pipeline integration (end-to-end, slow)
# ==============================================================
@pytest.mark.slow
class TestIntegration:

    def test_transcription_produces_text(self):
        """Whisper should produce a non-trivial transcript from the sample audio."""
        if not os.path.exists(fpl.AUDIO_FILE):
            pytest.skip(f"Audio file not found: {fpl.AUDIO_FILE}")
        transcript = fpl.transcribe_local_audio(fpl.AUDIO_FILE)
        assert transcript is not None
        assert len(transcript) > 100

    def test_full_pipeline_runs(self):
        """End-to-end pipeline runs without crashing."""
        if not os.path.exists(fpl.AUDIO_FILE):
            pytest.skip(f"Audio file not found: {fpl.AUDIO_FILE}")

        players, teams = fpl.fetch_fpl_data_direct()
        assert players is not None

        player_index = fpl.build_player_index(players)
        buckets = fpl._bucket_index(player_index)

        audio_summary = fpl.analyze_audio(fpl.AUDIO_FILE, player_index, buckets)
        assert isinstance(audio_summary, str)
        assert len(audio_summary) > 0

        audio_players = fpl.find_players_with_context(
            audio_summary, player_index, buckets
        )
        players_detected = sorted({p["name"] for p in audio_players})

        sentiment, overall = fpl.get_sentiment_for_players(players_detected)
        assert isinstance(sentiment, dict)
        assert isinstance(overall, dict)

        top, full, mse, err = fpl.predict_fpl_points(players, teams)
        assert err is None

        decisions = fpl.orchestrate(
            audio_summary, audio_players, sentiment, top, full, teams
        )
        assert isinstance(decisions, list)