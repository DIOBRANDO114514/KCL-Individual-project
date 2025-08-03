# split_ycb_rgb.py 
import pathlib, random, shutil, os
random.seed(42)

SRC  = pathlib.Path('ycb_extract')      # Root directory where the object folder is located
DEST = pathlib.Path('ycb_rgb')          # Target ImageFolder root directory
for s in ('train', 'test'):
    (DEST/s).mkdir(parents=True, exist_ok=True)

for obj_dir in SRC.iterdir():           # target
    if not obj_dir.is_dir():
        continue

    # Find all png/jpg files — compatible with subdirectories containing rgb
    rgb_path = obj_dir/'rgb' if (obj_dir/'rgb').exists() else obj_dir
    imgs = sorted(rgb_path.glob('*.png')) + sorted(rgb_path.glob('*.jpg'))

    if len(imgs) == 0:                  # Some old packages may have few or no images.
        print(f'{obj_dir.name} No images found, skipping')
        continue

    random.shuffle(imgs)
    n = int(0.8 * len(imgs))
    splits = (('train', imgs[:n]), ('test', imgs[n:]))

    for split, paths in splits:
        out_dir = DEST / split / obj_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)
        for p in paths:
            shutil.copy2(p, out_dir / p.name)

print('Split complete, file written in ycb_rgb/train & ycb_rgb/test')