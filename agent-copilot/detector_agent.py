import pandas as pd
import glob
import os
import json
import ast

RESULTS_FILE = r"C:\Users\ADMIN\space-copilot\telemanom\results\full_benchmark_82_channels.csv"


def load_latest_full_results():
    df = pd.read_csv(RESULTS_FILE)
    if len(df) < 80:
        raise ValueError(f"Expected ~82 channels, found {len(df)} in {RESULTS_FILE}")
    return RESULTS_FILE, df

def detect_anomaly(channel_id=None):
    """
    Pick a real, caught anomaly from your actual results to investigate.
    If channel_id is given, use that specific channel. Otherwise pick
    the first channel with at least one true positive.
    """
    filepath, df = load_latest_full_results()

    if channel_id:
        row = df[df["chan_id"] == channel_id]
        if row.empty:
            raise ValueError(f"Channel {channel_id} not found in {filepath}")
        row = row.iloc[0]
    else:
        caught = df[df["true_positives"] > 0]
        if caught.empty:
            raise ValueError("No channels with true positives found.")
        row = caught.iloc[0]

    anomaly_class_raw = row["class"]
    # Strip brackets, split on comma, clean up whitespace/quotes
    cleaned = anomaly_class_raw.strip("[]")
    classes = [c.strip().strip("'").strip('"') for c in cleaned.split(",")]

    detection_summary = (
        f"Channel {row['chan_id']} ({row['spacecraft']}) had "
        f"{row['num_true_anoms']} labeled anomaly sequence(s) "
        f"of type(s) {classes}. The model correctly caught "
        f"{row['true_positives']}, missed {row['false_negatives']}, "
        f"and raised {row['false_positives']} false alarm(s)."
    )

    state_update = {
        "channel_id": row["chan_id"],
        "spacecraft": row["spacecraft"],
        "anomaly_class": classes[0] if classes else "unknown",
        "num_true_anomalies": int(row["num_true_anoms"]),
        "detection_summary": detection_summary,
    }
    return state_update


if __name__ == "__main__":
    result = detect_anomaly()
    print(json.dumps(result, indent=2))