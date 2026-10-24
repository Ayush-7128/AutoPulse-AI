"""Backend Inference & Maruti OEM Triage Service Module."""
import json
from backend.dsp_processor import bandpass_and_mel
from backend.rul_calculator import estimate_safe_drivable_km

def load_oem_catalog(path="backend/maruti_oem_catalog.json"):
    with open(path, "r") as f:
        return json.load(f)
