"""Basic demonstration tracker for the "Diffuseur" image sequences.

This tracker operates on the particle images as given by diffuseur_preproc.py.
In a given frame, it detects the particles and track them in the next frame
using a very basic cross-correlation approach.

Results are saved as a text file. Visualization on the fly is possible.

Written by P. DERIAN - www.pierrederian.net. 
"""

###
import os
import time
###
import matplotlib.pyplot as pyplot
import numpy
import scipy.ndimage as ndimage
import skimage.feature as feature
###

def track_particles(img0, img1, patchsize=[64, 32], mask=None,
                    min_radius=3, max_radius=10, threshold=15.,
                    upsample=10):
    """Detect particles in the 1st image and track them in the second.
    
    Arguments:
        - img0, img1: MxN scalar arrays (grayscale images).
        - mask=None: an optional MxN array of 0's and 1's, where areas to be
            excluded are denoted by 0.
        - patchsize=[64,32]: the (descending) sequence of patch size
            for the tracking.
        - min_radius=3, max_radius=5: the min, max radius [pixel] of particles [1]
        - threshold=15: the absolute threshold for particle detection, see [1]  
        - upsample=10: the upsample factor for tracking, see [2]        
    
    Returns: coords, shifts
        each a Px2 array for coordinates (y, x) [pixel] for the P particles
        detected and their associated displacement (uy, ux) [pixel].
        
    References:
        [1] http://scikit-image.org/docs/dev/api/skimage.feature.html#skimage.feature.blob_dog
        [2] http://scikit-image.org/docs/dev/api/skimage.feature.html#skimage.feature.register_translation
        
    Written by P. DERIAN 2016-11-25
    """
    # detect particles in 1st image using blob detection algorithm
    blobs = feature.blob_dog(img0, min_sigma=min_radius/numpy.sqrt(2.),
                             max_sigma=max_radius/numpy.sqrt(2.),
                             threshold=threshold)
    # dimensions
    dimy, dimx = img0.shape[:2] 
    maxPatchsize = max(patchsize)
    xmin = maxPatchsize
    xmax = dimx - maxPatchsize
    ymin = maxPatchsize
    ymax = dimy - maxPatchsize
    # for each blob
    coords = []
    shifts = []
    for b in blobs:
        # coordinates
        by0, bx0, bs = b
        # reject if too close to the border
        if (bx0<xmin) or (bx0>xmax) or (by0<ymin) or (by0>ymax):
            continue
        # or outside the mask
        if (mask is not None) and (not mask[int(by0), int(bx0)]):
            continue
        # initial motion
        ux, uy = 0, 0
        # for each patch
        for p in patchsize:
            by1 = by0 + int(uy)
            bx1 = bx0 + int(ux)
            hp = p/2            
            # patch
            patch0 = img0[int(by0-hp):int(by0+hp), int(bx0-hp):int(bx0+hp)]
            patch1 = img1[int(by1-hp):int(by1+hp), int(bx1-hp):int(bx1+hp)]
            # shift (upsample at finest resolution)
            s, _, _ = feature.register_translation(patch1, patch0, upsample_factor=(upsample if p==patchsize[-1] else 1))
            uy += s[0]
            ux += s[1]
        # reject if final destination is outside the mask
        if (mask is not None) and (not mask[int(by1), int(bx1)]):
            continue
        coords.append([by0, bx0])
        shifts.append([uy, ux])
    # to array
    coords = numpy.array(coords)
    shifts = numpy.array(shifts)
    return coords, shifts

def outliers_median_test(u, threshold=3.):
    """Finds the outliers of displacements by applying median test (MAD) [1].
    
    Arguments:
        - u: Px2 array of displacements uy, ux;
        - threshold=3: rejection threshold.
        
    Returns:
        - outliers, P logical array
            outliers[k]=1 <=> u[k,:] is an outlier. 
    
    Reference: 
        [1] https://en.wikipedia.org/wiki/Median_absolute_deviation
        
    Written by P. DERIAN 2016-11-28
    """
    # median vector
    mu = numpy.median(u, axis=0)
    # residuals |u - mu|
    res = numpy.sqrt(numpy.sum((u-mu)**2, axis=1))
    # median of residuals
    mres = numpy.median(res)
    return res>threshold*mres

def demo(outdir, display=True):
    """Demo for the tracker.
    
    Argument:
        - outdir: the output directory
        - display=True: enables or disables display of results.
    
    Written by P. DERIAN, 2016.
    """
    
    # data parameters
    patternEstim = '/Users/pderian/Documents/Data/SourcesHydro_ifremer/sequences-video-flux-p_rodier/rodier_seq5-MOMARSAT13_MOMARSAT1303_130828142725_15/diff/diff_{:04d}.jpg' # this is the pattern of particle images
    patternSource = '/Users/pderian/Documents/Data/SourcesHydro_ifremer/sequences-video-flux-p_rodier/rodier_seq5-MOMARSAT13_MOMARSAT1303_130828142725_15/{:04d}.jpg' # these are the raw images, for visualization purpose only (not used for tracking)
    fmask = '/Users/pderian/Documents/Data/SourcesHydro_ifremer/sequences-video-flux-p_rodier/rodier_seq5-MOMARSAT13_MOMARSAT1303_130828142725_15/mask.png' # this mask enables to exclude some areas.
    kmin = 200 #the first frame index to process 
    kmax = 1467 #the last one.
    
    # result files
    outfile = os.path.join(outdir, 'track.txt')
    with open(outfile, 'w') as of:
        # header
        of.write("#Tracking log\n")
        of.write("#Created by {} - {} \n".format(__file__, time.ctime()))
        of.write("#frame x[px] y[px] ux[px] uy[px]\n")
    
        # now begin estimation
        mask = pyplot.imread(fmask).astype('bool')
        for k in range(kmin,kmax-1):
            print k
            f0 = patternEstim.format(k)
            f1 = patternEstim.format(k+1)
            fs = patternSource.format(k)
            # load
            img0 = pyplot.imread(f0)
            img1 = pyplot.imread(f1)
            imgs = pyplot.imread(fs)
            img0_gray = numpy.average(img0, axis=-1)
            img1_gray = numpy.average(img1, axis=-1)

            # retrieve particles and displacements
            coords, shifts = track_particles(img0_gray, img1_gray, mask=mask)
            # quality control: median test
            outliers = outliers_median_test(shifts)
            inliers = numpy.logical_not(outliers)

            # display image
            if display:
                dimy, dimx = imgs.shape[:2] 
                dpi = 90.
                fig = pyplot.figure(figsize=(dimx/dpi, dimy/dpi))
                ax = fig.add_axes([0,0,1,1])
                ax.imshow(imgs)
                ax.quiver(coords[outliers,1], coords[outliers,0],
                  shifts[outliers,1], shifts[outliers,0],
                  color='r', units='xy', angles='xy', scale_units='xy',
                  scale=0.25, width=5)
                ax.quiver(coords[inliers,1], coords[inliers,0],
                  shifts[inliers,1], shifts[inliers,0],
                  color='y', units='xy', angles='xy', scale_units='xy',
                  scale=0.25, width=5)
                ax.set_xlim(0, dimx)
                ax.set_ylim(dimy, 0)
                pyplot.savefig(os.path.join(outdir, 'track_{:04d}.jpg'.format(k)), dpi=dpi)
                pyplot.close(fig)

            # write results for inliers only
            for i, is_valid in enumerate(inliers):
                if is_valid:
                    of.write("{:d} {:.1f} {:.1f} {:.1f} {:.1f}\n".format(
                             k,
                             coords[i,1], coords[i,0],
                             shifts[i,1], shifts[i,0]))


if __name__=="__main__":
    demo('/Users/pderian/Documents/Data/SourcesHydro_ifremer/sequences-video-flux-p_rodier/rodier_seq5-MOMARSAT13_MOMARSAT1303_130828142725_15/track', False)