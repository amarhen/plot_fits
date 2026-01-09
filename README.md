# plot_fits
Tool to plot one or more FITS files in raster/contour maps.

This simple code creates plots in matplotlib from FITS files. Multiple plots can be overlaid (e.g. raster + contours). Custom options include plotting the synthesized beam, scalebars, Regions, trimmed colormaps, masking, colored contours, etc.

Your FITS files must come from elsewhere. If you want to plot moment-zero images from a FITS cube, you have to produce it with another software, e.g. CASA/CARTA.

You will also need to measure the RMS noise of your image with one of these tools if you are interested in plotting contours.

Packages needed: astropy, numpy, matplotlib, Regions (https://pypi.org/project/regions/).

This code was employed to create the plots in Martínez-Henares et al. (2025) https://ui.adsabs.harvard.edu/abs/2025A%26A...699A.382M/abstract . If you use this code for your research, a mention in your publication is much appreciated.

An example of the type of images that can be created is shown below.

![](https://github.com/amarhen/plot_fits/blob/main/example.png?raw=true)

## How to use the code

To use this code, first the options have to be edited in the code, and then run.

The `plot_fits.py` file is the code itself, and it includes a working example of how it should be used. The best way to make sure that you can use the code is to reuse the example I provide with your own files and custom options. 

The workflow to use this code is:

1. Set the coordinates of the center of the plot.
2. Set the parameters of the plot: axes, font size, color scale, scalebar, masked values, ...
3. Open a FITS file
4. Create plot(s) for this FITS file
5. Repeat the last two steps for as many FITS files you want to include
6. Gather all plots in the final image

When all options are set, you can run the code by

```
python plot_fits.py
```

### 1. Setting coordinates

Edit parameters `raref` and `decref` in lines 520-521.

### 2. Setting plot parameters

Edit the dictionary `plotparams` from line 523. The example code includes all possible options.

### 3. Open a FITS file

Each FITS file is an instance of the `FITSmap` class. To create one, declare it as in line 551. You will need one per FITS file used.

### 4. Create plots

Once the `FITSmap` instance is created, it is used to create one or more `MapForPlot` instances, which will be the very objects to be plotted onto the final image. The `MapForPlot` object represents the raster/contour plot with all the custom options. You can see an example of a raster and a contour plot from an SiO moment-zero FITS file in lines 564-586. 

### 5. Repeat

You can add as many `MapForPlot` instances (i.e. plots) from as many `FITSmap` (i.e. files) as you want. Of course, at some point the image will be too crowded. Note also that it only makes sense to plot one raster map, while you can have multiple contour maps.

### 6. Gather all plots in the final image

Create a list of all the `MapForPlot` instances, as in the `mapforplots_list` of the example code in line 623. Then use the function `plot_maps` to effectively plot all together in a final image. You can also add a Region (see lines 612-615).

## Potential issues

This code was developed for a very specific case. Therefore, if the plot parameters (e.g. the size) change too much, visual elements such as the beam ellipse or the scalebar might not be totally alligned with their background shadow. To solve this, some level of user editing would be necessary. These parameters are set in lines 370-437.

## Bugs, questions

Feel free to contact me at amartinez@cab.inta-csic.es . 
