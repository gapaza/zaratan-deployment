import os

#
#       _____   _                   _                _
#      |  __ \ (_)                 | |              (_)
#      | |  | | _  _ __  ___   ___ | |_  ___   _ __  _   ___  ___
#      | |  | || || '__|/ _ \ / __|| __|/ _ \ | '__|| | / _ \/ __|
#      | |__| || || |  |  __/| (__ | |_| (_) || |   | ||  __/\__ \
#      |_____/ |_||_|   \___| \___| \__|\___/ |_|   |_| \___||___/
#
# 1.3897, 1.3897, 0.2148, 0.1696, 0.1436, 0.1261, 0.1130, 0.1028, 0.0946, 0.0879, 0.0773


parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
root_dir = os.path.join(parent_dir, 'zaratan-deployment')


plots_dir = os.path.join(root_dir, 'plots')
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)


#      __  __           _      _
#     |  \/  |         | |    | |
#     | \  / | ___   __| | ___| |
#     | |\/| |/ _ \ / _` |/ _ \ |
#     | |  | | (_) | (_| |  __/ |
#     |_|  |_|\___/ \__,_|\___|_|

batch_size = 16

IMG_CHANNELS = 1  # Density field
IMG_RES = 64
IMAGE_SHAPE = (IMG_RES, IMG_RES)
TIME_DIM = 32

# Denoising Schedule
TIME_STEPS = 1000
TIME_STEPS_DIMM = 5
BETA_START = 1e-4
BETA_END = 0.02  # 0.02
CLIP_MIN = -1.0
CLIP_MAX = 1.0

# Fine Tuning
FT_TIME_STEPS = 1000
LOG_PROB_STD_MIN = 0.1  # So far fine-tuning works best at 0.01, paper uses 0.1









