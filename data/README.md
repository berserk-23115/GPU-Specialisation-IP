# Sample Data Directory

This directory is meant to contain sample images for processing. You should place your images in the following structure:

```
data/
├── input/               # Place input images here
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── output/              # Processed images will be saved here
└── README.md            # This file
```

## Sample Images

For testing the application, you should use:
- At least 10 small images (less than 1 MB each) for batch processing tests
- At least 2 large images (more than 5 MB each) for performance tests
- Images with different color depths (grayscale, RGB, RGBA)
- Different image formats (JPEG, PNG, BMP)

## Creating Input Directory

To set up the directory structure:

```bash
mkdir -p data/input data/output
```

Then place your images in the input directory. 