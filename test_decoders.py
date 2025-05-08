import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from vis.glance import GlanceWriterDecoder
from vis.no_delete_processing import process_participant_data



all_participant_data = []
for participant_num in range(1, 8):  # Changed from range(1, 10) to range(1, 12) to include participants 10 and 11
    try:
        participant_data = process_participant_data(participant_num)
        all_participant_data.append(participant_data)
    except FileNotFoundError:
        continue

print(all_participant_data)