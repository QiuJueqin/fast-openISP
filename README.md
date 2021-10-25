# Fast Open Image Signal Processor (fast-openISP)

As told by its name, fast-openISP is a **faster** re-implementation of the [openISP](https://github.com/cruxopen/openISP) project.

Compared to C-style code in the official openISP repo, fast-openISP uses pure matrix implementations based on Numpy, and increases processing speed over xxx.

Here is the running time in my Ryzen 7 1700 8-core 3.00GHz machine:

|Camera   |File                                      |Raw Resolution|openISP Pipeline Time|fast-openISP Pipeline Time|
|---------|------------------------------------------|--------------|---------------------|--------------------------|
|unknown  |[test.RAW](raw/test.RAW)                  |1920 x 1080   |todo                 |todo                      |
|Nikon D3x|[color_checker.pgm](raw/color_checker.pgm)|6080 x 4044   |todo                 |todo                      |


# Algorithms

All modules in fast-openISP reproduce processing algorithms in openISP, except for EEH and BCC modules.

### EEH (edge enhancement)

The official openISP uses the difference between the original and the gaussian-filtered Y-channel arrays as the edge map approximation. In fast-openISP, however, we replace gaussian with the bilateral filter, which gets better estimation to the edges, and consequently reduces the artifact when the enhancement gain is large.  

### BCC (brightness & contrast control)

The official openISP enhances the image contrast by pixel-wise enlarge the difference between pixel values and a constant integer (128). In fast-openISP, we use the median value of the whole frame instead of a constant.


# Parameters

Tunable module parameters in fast-openISP are differently named from those in openISP, but they are all self-explained, and no doubt you can easily tell the counterparts in two repos. All parameters are managed in a yaml configuration file, and one yaml for one camera.   

# Demo

`# todo`


# License

Copyright 2021 Qiu Jueqin.

Licensed under [MIT](http://opensource.org/licenses/MIT).
