# CS-GMM
This repository contains our implementation and report on compressed sensing based solutions for image inpainting and interpolative zooming using Gaussian Mixture Models, completed for a course project in CS 754 - Advanced Image Processing, in Spring 2021.

This work is based on the following paper: https://arxiv.org/pdf/1006.3056.pdf.

Authors: Ankit Kumar Misra (190050020) and Dhruva Dhingra (190070020).

Directory structure:
- `report.pdf` is a report on the mathematics behind this method and the performance results of our implementations.
- `code` contains MATLAB programs that implement image inpainting and interpolative zooming techniques using Gaussian Mixture Models. It also contains code for a special initialization of the GMM parameters, as per the paper linked above.
- `data` contains the three images on which our implementations were tested to analyze performance.
- `results` contains inpainted and zoomed image results generated by our programs, for all three images in `data`.

When running the MATLAB programs, do make sure they are present in the same directory, since some of the files are inter-dependent.
