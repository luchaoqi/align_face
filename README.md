forked from: https://gist.github.com/lzhbrian/bde87ab23b499dd02ba4f588258f57d5

may also check other references:  
https://github.com/yuval-alaluf/hyperstyle/blob/main/scripts/align_faces_parallel.py

todo:
add `only_keep_largest` from https://github.com/sczhou/CodeFormer/blob/master/scripts/crop_align_face.py

requirements:
```
pip install numpy pillow dlib scipy
wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
bzip2 -dk shape_predictor_68_face_landmarks.dat.bz2
ln -sf /usr/lib/x86_64-linux-gnu/libstdc++.so.6 ${CONDA_PREFIX}/lib/libstdc++.so.6
```

usage: `python align_face_multi.py --input_dir /path/to/img_dir`, you may adjust the number of cores for multi-processing.

For in-code face alignment (no separate data preprocessing through CLI), refer to `facexlib` where an example in GFPGAN is given [here](https://github.com/TencentARC/GFPGAN/blob/7552a7791caad982045a7bbe5634bbf1cd5c8679/gfpgan/utils.py#L79-L148).
> sanity check of facexlib code justifies it with the following links: [face_template values · Issue #14 · xinntao/facexlib](https://github.com/xinntao/facexlib/issues/14) and [source code](https://github.com/xinntao/facexlib/blob/e5768d1722a3fddc6ccd1b91a6a17f432ed149b4/facexlib/utils/face_restoration_helper.py#L68).
