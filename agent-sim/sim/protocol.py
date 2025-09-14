"""
Negotiation protocol engine that orchestrates the negotiation process.
Handles different protocols: alternating, simultaneous, random.
"""

import random
import time
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

from models import (
    Entity, Issue, Offer, OfferStatus, NegotiationRound,
    NegotiationOutcome, SimulationConfig
)
