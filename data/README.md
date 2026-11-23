# Dataset Directory Format (MSTAR 3-Class SAR)

Raw MSTAR images are excluded from version control to keep the repository lightweight.
To run training or inference locally, organize your dataset as follows:

- `data/train.txt`: Each line contains `<relative_image_path> <integer_label>` (`0`: BMP2, `1`: BTR70, `2`: T72).
- `data/test.txt`: Each line contains `<relative_image_path>` for unlabeled test samples.
