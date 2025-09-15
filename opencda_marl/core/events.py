'''
Author       : AXIBA leolihao@arizona.edu
Date         : 2025-09-05 15:46:15
FilePath     : /OpenCDA-MARL/opencda_marl/core/events.py
Description  : 
Copyright (c) 2025 by AXIBA (leolihao@arizona.edu), All Rights Reserved.
'''
from dataclasses import dataclass


@dataclass
class StepEvent:
    step: int
    event_id: str
    vehicle_id: int
    event_type: str
