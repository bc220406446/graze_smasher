"""
Graze Smasher Product Hunting Tool - Rule-Based IP Velocity Detector
CS619 Final-Year Project

Detects suspicious review velocity: flags an IP address if more than N reviews
are submitted within a configurable time window.
"""

import time
import threading
from collections import defaultdict
import numpy as np
import pandas as pd

try:
    from .config import IP_MAX_REVIEWS_PER_WINDOW, IP_TIME_WINDOW_MINUTES
except ImportError:
    from config import IP_MAX_REVIEWS_PER_WINDOW, IP_TIME_WINDOW_MINUTES


class IPVelocityTracker:
    """
    Thread-safe in-memory sliding-window velocity tracker for IP addresses.
    Tracks review submission timestamps per IP and flags bursts exceeding limits.
    """
    def __init__(self, max_reviews: int = IP_MAX_REVIEWS_PER_WINDOW, window_minutes: int = IP_TIME_WINDOW_MINUTES):
        self.max_reviews = max_reviews
        self.window_seconds = window_minutes * 60
        self.history = defaultdict(list)
        self.lock = threading.Lock()

    def check_ip(self, ip: str | None, current_time: float | None = None, record: bool = False) -> tuple[bool, str]:
        """
        Checks if an IP has exceeded the allowed review submission threshold.
        
        Parameters:
            ip (str | None): IP address to inspect
            current_time (float | None): Epoch timestamp (defaults to time.time())
            record (bool): Whether to record this submission in history
            
        Returns:
            tuple[bool, str]: (is_flagged, description)
        """
        if not ip:
            return False, "No IP provided"
            
        now = current_time if current_time is not None else time.time()
        window_start = now - self.window_seconds
        
        with self.lock:
            # Filter out entries older than window
            timestamps = [t for t in self.history[ip] if t >= window_start]
            self.history[ip] = timestamps
            
            # Check count
            is_flagged = len(timestamps) >= self.max_reviews
            
            if record:
                self.history[ip].append(now)
                
            if is_flagged:
                reason = (f"Suspicious activity: IP {ip} submitted {len(timestamps)} reviews "
                          f"within {self.window_seconds // 60} minutes (threshold: {self.max_reviews}).")
            else:
                reason = "IP velocity within normal limits"
                
            return is_flagged, reason

    def record_submission(self, ip: str | None, current_time: float | None = None) -> bool:
        """Records a review submission for the given IP. Returns True if now flagged."""
        flagged, _ = self.check_ip(ip, current_time=current_time, record=True)
        return flagged

    def clear(self):
        """Clears all tracking history (useful in tests)."""
        with self.lock:
            self.history.clear()


# Default singleton instance
tracker = IPVelocityTracker()


def check_ip_velocity(ip: str | None, record: bool = False) -> tuple[bool, str]:
    """Convenience wrapper for global IP velocity tracker."""
    return tracker.check_ip(ip, record=record)


def generate_simulated_ips(num_records: int, seed: int = 42) -> list[str]:
    """
    Generates a synthetic list of IP addresses for demo/testing purposes.
    The real dataset contains no IP addresses, so this provides clearly labelled
    simulated IPs, including benign unique addresses and burst spammer addresses.
    
    Parameters:
        num_records (int): Total number of rows needing IPs
        seed (int): Random seed for reproducible assignment
        
    Returns:
        list[str]: Array of simulated IP address strings
    """
    np.random.seed(seed)
    
    # 5 known spammer IPs that generate burst traffic
    spammer_ips = [
        "198.51.100.11",
        "198.51.100.22",
        "198.51.100.33",
        "198.51.100.44",
        "198.51.100.55"
    ]
    
    # Pool of 500 benign regular IPs
    benign_ips = [f"203.0.113.{i}" for i in range(1, 255)] + [f"192.0.2.{i}" for i in range(1, 246)]
    
    simulated_ips = []
    for i in range(num_records):
        # 8% probability of being from a spammer cluster
        if np.random.rand() < 0.08:
            simulated_ips.append(np.random.choice(spammer_ips))
        else:
            simulated_ips.append(np.random.choice(benign_ips))
            
    return simulated_ips

