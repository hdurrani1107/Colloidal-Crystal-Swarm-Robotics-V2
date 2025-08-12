# metrics.py
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import csv
import os

@dataclass
class MetricsLogger:
    """
    Minimal metrics logger for timeseries (one row per frame) and discrete events
    (zone_spawn, first_arrival, zone_complete).
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
        num_zones: int,
        total_trapped: int,
        mean_trapped: float,
        arrivals_this_frame: int,
    ):
        self.frame_rows.append({
            "frame": frame,
            "time": t,
            "system_temp": system_temp,
            "num_zones": num_zones,
            "total_trapped": total_trapped,
            "mean_trapped": mean_trapped,
            "arrivals": arrivals_this_frame,
        })

    # --- event logging ---
    def log_event(
        self,
        frame: int,
        t: Optional[float],
        event: str,             # "zone_spawn" | "first_arrival" | "zone_complete"
        zone_id: int,
        zone_pos_x: float,
        zone_pos_y: float,
        zone_radius: float,
        trapped_count: int,
        extra: Optional[Dict[str, Any]] = None,
    ):
        row = {
            "frame": frame,
            "time": t,
            "event": event,
            "zone_id": zone_id,
            "zone_pos_x": zone_pos_x,
            "zone_pos_y": zone_pos_y,
            "zone_radius": zone_radius,
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
                all_fieldnames = set()
                for row in self.frame_rows:
                    all_fieldnames.update(row.keys())
                writer = csv.DictWriter(f, fieldnames=sorted(all_fieldnames))
                writer.writeheader()
                writer.writerows(self.frame_rows)

        # Events
        ev_path = os.path.join(self.out_dir, "events.csv")
        if self.event_rows:
            with open(ev_path, "w", newline="") as f:
                all_fieldnames = set()
                for row in self.event_rows:
                    all_fieldnames.update(row.keys())
                writer = csv.DictWriter(f, fieldnames=sorted(all_fieldnames))
                writer.writeheader()
                writer.writerows(self.event_rows)