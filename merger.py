import sys
from os.path import isfile, isdir, join
from os import makedirs, listdir, remove
import shutil
import numpy as np
import numpy.linalg as la
import json
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

MAX_RANGE = 300

dataset_dir = sys.argv[1]
assert isdir(dataset_dir), dataset_dir
print('merge ', dataset_dir)

DATASET_FILE = join(dataset_dir, 'dataset.json')
with open(DATASET_FILE, 'r') as f:
    DATASET = json.load(f)

ALL_FILES = [join(dataset_dir, f) for f in listdir(dataset_dir) if f.endswith('_unroll.json')]
assert len(ALL_FILES) == 1, str(ALL_FILES)
ANNOTATION_FILE = ALL_FILES[0]

with open(ANNOTATION_FILE, 'r') as f:
    ANNOTATIONS = json.load(f)

VALID_FRAMES = set(DATASET['valid_frames'])
scale_to_mm = DATASET['scale_to_mm']

PERSON_FRAME_2_LOCATION = {}
FRAME_2_PERSONS = {}

for entry in ANNOTATIONS['persons']:
    frame = entry['frame']
    if frame in VALID_FRAMES:
        if frame not in FRAME_2_PERSONS:
            FRAME_2_PERSONS[frame] = []
        for person in entry['persons']:
            pid = person['pid']
            location = np.squeeze(person['location'])
            if len(location) == 3:
                FRAME_2_PERSONS[frame].append(pid)
                PERSON_FRAME_2_LOCATION[pid, frame] = location[0:2] * scale_to_mm  # we only care about the point on the floor

print('valid frames', len(VALID_FRAMES))

TRACK_DIR = join(dataset_dir, 'tracks3d')
assert isdir(TRACK_DIR), TRACK_DIR

TRACK_BACKUP_DIR = join(dataset_dir, 'tracks3d_backup')
if isdir(TRACK_BACKUP_DIR):
    print('[system is already merged!]')
    exit(0)

FRAME_2_POSES = {}
FRAME_PERSON_2_POSE = {}

J = 24
z_axis = 2

for track_file in [join(TRACK_DIR, f) for f in listdir(TRACK_DIR) if f.endswith('json')]:
    with open(track_file, 'r') as f:
        track = json.load(f)
        for frame, pose in zip(track['frames'], track['poses']):
            if frame not in FRAME_2_POSES:
                FRAME_2_POSES[frame] = []
            FRAME_2_POSES[frame].append(pose)

for frame in tqdm(VALID_FRAMES):
    persons = FRAME_2_PERSONS[frame]
    poses = FRAME_2_POSES[frame]
    if len(persons) > 0 and len(poses) > 0:

        n = len(poses)
        m = len(persons)
        C = np.zeros((n, m))

        for i, pose in enumerate(poses):
            pts = []
            for pt in pose:
                if pt is not None:
                    pts.append(pt)
            pts = np.mean(pts, axis=0) * scale_to_mm
            pose_pt = pts[0:2]  # project point to floor

            for j, pid in enumerate(persons):
                person_pt = PERSON_FRAME_2_LOCATION[pid, frame]

                d = la.norm(pose_pt - person_pt)

                C[i, j] = d

        row_ind, col_ind = linear_sum_assignment(C)
        for i, j in zip(row_ind, col_ind):
            d = C[i, j]
            if d < MAX_RANGE:
                pose = poses[i]
                pid = persons[j]
                FRAME_PERSON_2_POSE[frame, pid] = pose

OUTPUT = []
for (frame, pid), pose in FRAME_PERSON_2_POSE.items():
    OUTPUT.append({
        'frame': frame,
        'pid': pid,
        'pose': pose
    })

OUTPUT_FILE = join(dataset_dir, 'persons2poses.json')
if isfile(OUTPUT_FILE):
    remove(OUTPUT_FILE)

with open(OUTPUT_FILE, 'w') as f:
    json.dump(OUTPUT, f)
