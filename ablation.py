"""
Ablation study: quantify each model's contribution to the orchestrator.

This runs the pipeline ONCE to collect all intermediate outputs, then
re-runs the orchestrator with different subsets of those outputs.
This isolates the orchestrator's decision logic from the slow
Whisper / Llama / Reddit stages.

Run with:
    python ablation.py
"""

import prototype_fpl as fpl


def run_ablation():
    print("=" * 60)
    print("🧪 ABLATION STUDY — ORCHESTRATOR DECISION CONTRIBUTIONS")
    print("=" * 60)

    # ----------------------------------------------------------
    # 1. Run the expensive stages ONCE and cache their outputs
    # ----------------------------------------------------------
    print("\n🔄 Running full pipeline once to cache model outputs...")
    players, teams = fpl.fetch_fpl_data_direct()
    if players is None:
        print("❌ Failed to fetch FPL data. Aborting.")
        return

    player_index = fpl.build_player_index(players)
    buckets = fpl._bucket_index(player_index)

    audio_summary = fpl.analyze_audio(fpl.AUDIO_FILE, player_index, buckets)
    audio_players = fpl.find_players_with_context(
        audio_summary, player_index, buckets
    )
    players_detected = sorted({p["name"] for p in audio_players})
    print(f"   Audio-detected players: {players_detected}")

    sentiment_full, overall = fpl.get_sentiment_for_players(players_detected)
    print(f"   Reddit sentiment collected for {len(sentiment_full)} players.")

    top_predictions, full_df, mse, err = fpl.predict_fpl_points(players, teams)
    if err:
        print(f"❌ ML model failed: {err}")
        return
    print(f"   ML model MSE = {mse:.3f}")

    # ----------------------------------------------------------
    # 2. Define ablation configurations
    # ----------------------------------------------------------
    # Each config disables one or more model outputs before
    # passing them to orchestrate(). This is equivalent to
    # asking: "How many actionable decisions would we produce
    # if we had NO audio / NO sentiment / NO ML?"
    # ----------------------------------------------------------
    configs = [
        ("All models",            True,  True,  True),
        ("No audio",              False, True,  True),
        ("No sentiment",          True,  False, True),
        ("No ML model",           True,  True,  False),
        ("Audio only",            True,  False, False),
        ("Sentiment only",        False, True,  False),
        ("ML only",               False, False, True),
    ]

    results = []
    for label, use_audio, use_sent, use_ml in configs:
        ap = audio_players if use_audio else []
        sent = sentiment_full if use_sent else {}
        top = top_predictions if use_ml else top_predictions.iloc[0:0]
        full = full_df if use_ml else full_df

        decisions = fpl.orchestrate(
            audio_summary, ap, sent, top, full, teams
        )
        actionable = [
            d for d in decisions
            if d["decision"] in ("AVOID", "TRANSFER IN", "CONSIDER")
        ]
        results.append({
            "Configuration": label,
            "Total decisions": len(decisions),
            "Actionable decisions": len(actionable),
            "Actionable decisions detail": ", ".join(
                f"{d['name']}({d['decision']})" for d in actionable
            ) or "—",
        })

    # ----------------------------------------------------------
    # 3. Report
    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("📊 ABLATION RESULTS")
    print("=" * 60)

    for r in results:
        print(f"\n▶ {r['Configuration']}")
        print(f"    Total decisions:      {r['Total decisions']}")
        print(f"    Actionable decisions: {r['Actionable decisions']}")
        print(f"    Detail:               {r['Actionable decisions detail']}")

    # ----------------------------------------------------------
    # 4. Interpretation
    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("📝 INTERPRETATION")
    print("=" * 60)

    full_row = results[0]
    full_actionable = full_row["Actionable decisions"]
    print(f"\nFull pipeline produced {full_actionable} actionable decision(s).")

    for r in results[1:]:
        delta = r["Actionable decisions"] - full_actionable
        if delta == 0:
            verdict = "no loss — this model did not contribute"
        elif delta < 0:
            verdict = f"lost {abs(delta)} actionable decision(s)"
        else:
            verdict = f"gained {delta} decision(s) unexpectedly"
        print(f"  Removing {r['Configuration']:15s}: {verdict}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    run_ablation()