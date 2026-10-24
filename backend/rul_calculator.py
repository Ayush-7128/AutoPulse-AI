"""Remaining Useful Life (RUL) & Safe Drivable Horizon Calculator for Maruti Fleet."""
BASE_HORIZON_KM = {0: 10000, 1: 15, 2: 60, 3: 10000, 4: 140, 5: 25, 6: 45, 7: 10000, 8: 20}

def estimate_safe_drivable_km(class_id, confidence, kurt_z=3.0):
    base_km = BASE_HORIZON_KM.get(class_id, 100)
    if base_km >= 10000:
        return base_km
    severity_penalty = max(0.5, min(1.2, 3.0 / max(kurt_z, 2.5)))
    return int(base_km * severity_penalty * (1.0 - 0.15 * confidence))
