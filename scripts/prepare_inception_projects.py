import hashlib
import json
import shutil

import more_itertools

from qanno.paths import PATH_DATA_PDFS_CHUNKED, PATH_DATA_PDFS_SELECTED, PATH_ROOT

BATCH_SIZE = 40

PATH_DATA_PDFS_CHUNKED.mkdir(exist_ok=True, parents=True)

# 01 89ff5e47307c50223dc12006d92aeda5 50
# 02 1287cf325ad075ba87cad4ea24c1dc12 50
# 03 259e694598cbf9da4283f75b8bbc5a5e 50
# 04 bae70085023847073cb277fa1ebd1dc1 50
# 05 f5345746d6c7b4c511d02b944687a932 50
# 06 51af1e63b628ae9cc84161b83c909cbc 50
# 07 46a5962dc11b529be8e7d8d1bcdbb39e 50
# 08 8545aa049125f4ae64304c1feb8f2d02 50
# 09 4f60359cdbc015c21a81828f27bd7310 50
# 10 8ddb66174620ae6d9d0b56904b76b635 50
# 11 f940c5a4b539da82598f96b964517bff 50
# 12 5c9c26d61309ed53aaf60312df3e6310 50
# 13 1476207272fe92558cbdd65f2c28911d 50
# 14 8454f0d3bfcdeb2ec898528ebd451a8f 40

# Clean up

batches = {}

for d in PATH_DATA_PDFS_CHUNKED.iterdir():
    for f in d.iterdir():
        if f.is_file():
            f.unlink()

sorted_papers = list(sorted(PATH_DATA_PDFS_SELECTED.iterdir(), key=lambda x: x.name))
for i, chunk in enumerate(more_itertools.chunked(sorted_papers, BATCH_SIZE)):
    dest_dir = PATH_DATA_PDFS_CHUNKED / f"{i + 1:02}"
    dest_dir.mkdir(exist_ok=True, parents=True)

    h = hashlib.new("md5")

    batch = []

    for f in chunk:
        assert f.is_file()
        assert f.suffix == ".pdf"

        dest_path = dest_dir / f.name.replace("+", "").replace("&", "")

        h.update(dest_path.name.encode("utf-8"))

        shutil.copy(f, dest_path)

        batch.append(f.name)

    idx = f"{i + 1:02}"
    batches[idx] = batch

    print(f"# {idx} {h.hexdigest()} {len(batch)}")

with open(PATH_ROOT / "batches.json", "w") as f:
    json.dump(batches, f, indent=2)
