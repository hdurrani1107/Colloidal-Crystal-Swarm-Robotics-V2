# metrics.py
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import csv
import os

@dataclass
class MetricsLogger:
    """
    Minimal metrics logger for timeseries (one row per frame) and discrete events
    (beacon_spawn, first_arrival, beacon_complete).
    """
    out_dir: str
    frame_rows: List[Dict[str, Any]] = field(default_factory=list)
    event_rows: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self):
        os.makedirs(self.out_dir, exist_ok=True)

    # --- per-frame logging ---
    def log_frame(
        self,
        frame: int,
        t: float,
        system_temp: float,
        num_beacons: int,
        total_trapped: int,
        mean_trapped: float,
        arrivals_this_frame: int,
    ):
        self.frame_rows.append({
            "frame": frame,
            "time": t,
            "system_temp": system_temp,
            "num_beacons": num_beacons,
            "total_trapped": total_trapped,
            "mean_trapped": mean_trapped,
            "arrivals": arrivals_this_frame,
        })

    # --- event logging ---
    def log_event(
        self,
        frame: int,
        t: Optional[float],
        event: str,             # "beacon_spawn" | "first_arrival" | "beacon_complete"
        beacon_id: int,
        beacon_pos_x: float,
        beacon_pos_y: float,
        beacon_radius: float,
        trapped_count: int,
        extra: Optional[Dict[str, Any]] = None,
    ):
        row = {
            "frame": frame,
            "time": t,
            "event": event,
            "beacon_id": beacon_id,
            "beacon_pos_x": beacon_pos_x,
            "beacon_pos_y": beacon_pos_y,
            "beacon_radius": beacon_radius,
            "trapped_count": trapped_count,
        }
        if extra:
            row.update(extra)
        self.event_rows.append(row)

    def save(self):
        # Timeseries
        ts_path = os.path.join(self.out_dir, "timeseries.csv")
        if self.frame_rows:
            with open(ts_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(self.frame_rows[0].keys()))
                writer.writeheader()
                writer.writerows(self.frame_rows)

        # Events
        ev_path = os.path.join(self.out_dir, "events.csv")
        if self.event_rows:
            with open(ev_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(self.event_rows[0].keys()))
                writer.writeheader()
                writer.writerows(self.event_rows)