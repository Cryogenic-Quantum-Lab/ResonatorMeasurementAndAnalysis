# -*- coding: utf-8 -*-
"""
Shared configuration for the resonator measurement + analysis scripts.

This file has no hardware dependencies, so it's safe to import from
measurement.py, analysis.py, or main.py without side effects.

Note: the Dependencies folder location is set as DEPENDENCY_DIR in
measurement.py (and mirrored in analysis.py), not here — it's a one-time
setup value, not something you need to change per experiment.
"""

# =============================================================================
# CONFIG: CHANGE THESE INPUT PARAMETERS
# =============================================================================
res_freq = [6.398372E9]  # these are centers of resonances
phase_offset = [360 - 252]

tau_split_count = 51  # essentially the num of segments
assumed_Qtot = [600]  # together with tau split count determines our calculations for the spread of our segmented sweep

bandwidth = 0.01  # in khz

edelay = 60
magn_slope = [4.35]
magn_offset = 0

average = [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
           15**2, 20**2, 25**2, 30**2, 35**2, 40**2]  # Should be 17 power levels, so 17 averaging levels

directory = r"C:\Users\cryoq\Documents\ResonatorData"

# Do not change values below unless you understand how power level changes
attn = 17
power_sequence = [5, 0, -5, -10, -15, -20, -25, -30, -35, -40,  # First batch at attn=17
                   -5, -10, -15, -20, -25, -30, -35]             # Second batch at attn=60
# =============================================================================


def check_parameter_lengths():
    """Sanity-check that the per-resonator / per-power-level lists line up."""
    if len(res_freq) == len(phase_offset) == len(assumed_Qtot) == len(magn_slope) \
            and len(average) == len(power_sequence):
        print("Input parameter lengths are consistent.")
        return True
    else:
        print("Error: Inconsistent input parameter lengths.")
        return False
