import pathlib, tqdm
from PIL import Image

SRC  = pathlib.Path('ycb_rgb')          # train/test root
DST  = pathlib.Path('ycb_rgb256')       # out put
EXTS = ('*.png', '*.PNG', '*.jpg', '*.JPG')
SIZE = 256

for split in ('train', 'test'):
    files = []
    for ext in EXTS:                    # Collect all upper and lower case png/jpg files.
        files.extend((SRC/split).rglob(ext))

    if not files:
        print(f'⚠️ {split} No images found. Check the path or extension.')
        continue

    for img_path in tqdm.tqdm(files, desc=f'{split} resize→jpg'):
        rel = img_path.relative_to(SRC)
        out_path = DST / rel.with_suffix('.jpg')
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with Image.open(img_path) as im:
            im = im.convert('RGB')
            im.thumbnail((SIZE, SIZE), Image.LANCZOS)
            im.save(out_path, 'JPEG', quality=90, optimize=True)
