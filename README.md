# Raw Image Decoder

## Description

Decodes .DNG files applying pixel interpolation (demosicing), white balance, and gamma correction.

## Usage

Shows help:

```python
python decode.py --help
```

Example of input processing:

```python
python decode.py -i scene.dng -w (621,2086) -o results
```
