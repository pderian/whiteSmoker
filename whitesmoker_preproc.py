"""Preprocessor for the "Diffuseur" image sequences.

The goal here is to isolate the moving particles from the relatively stationary
background. It uses the "Running Gaussian Average" to perform foreground detection.

Written by P. DERIAN - www.pierrederian.net. 
"""
###
import os.path
import subprocess
###
import matplotlib.pyplot as pyplot
import numpy
import skimage.exposure as exposure
import skimage.morphology as morphology
###

### configuration
IMG_DIR = '/Users/pderian/Documents/Data/SourcesHydro_ifremer/sequences-video-flux-p_rodier/rodier_seq5-MOMARSAT13_MOMARSAT1303_130828142725_15'
DIFF_DIR = os.path.join(IMG_DIR, 'diff')
IMG_PATTERN = '{:04d}.jpg'
DIFF_PATTERN = 'diff_{:04d}.jpg'
KMIN = 200 #first frame of the sequence to be processed
KMAX = 1468 #last frame (excluded)
KWIN = 10 #running window length (in frame)

class RunningGaussianAverage:
    """
    Adapted from https://en.wikipedia.org/wiki/Background_subtraction#Running_Gaussian_average
    
    Written by P. DERIAN, 2016.
    """
    
    def __init__(self, mu=None, sigma=1.):
        """Initialize with mean (mu) and standard deviation (sigma).
        
        Arguments:
            - mu, sigma: the mean and std dev, either constant scalar or images.
        """
        self.mu = mu
        self.sigmasqr = sigma**2
    
    def update(self, img, rho=0.01):
        """Update mean and standard deviation with new image.
        
        Arguments: 
            - img: the new image;
            - rho: its weight.
        """
        # the mean
        if self.mu is None:
            self.mu = img
        else:
            self.mu = rho*img + (1. - rho)*self.mu
        # the variance
        d = numpy.abs(img - self.mu)
        self.sigmasqr = rho*(d**2) + (1. - rho)*self.sigmasqr
    
    def classify(self, img, threshold=2.5):
        """Classify each pixel as bg (False) or fg (True)
        
        Arguments:
            - img: the image to classify according to current background;
            - threhsold: the classification threshold.
            
        Return: a binary image, True where pixels are classified as foreground.
        
        Note: if the image is multichannel, the classification is performed on
              each channel.  
        """
        return (img-self.mu)/numpy.sqrt(self.sigmasqr) > threshold

    def update_classify(self, img, rho=0.01, threshold=2.5):
        """Update the statistics and classify the image
        """
        self.update(img, rho)
        return self.classify(img, threshold)    

def backgroundSubtraction(kmin, kmax, kwin=10,
                          indir=IMG_DIR, inpattern=IMG_PATTERN,
                          outdir=DIFF_DIR, outpattern=DIFF_PATTERN,
                          verbose=True):
    """Performs background subtraction on the image sequence, writes resulting
    foreground image.
    
    Arguments:
        - kmin, kmax: the min and max indices of the images in the sequence;
        - kwin: the size of the running average window;
        - indir, inpattern: the input directory and input images pattern;
        - outdir, outpattern: the input directory and output (foreground) images pattern;
        - verbose: enable/disable termnial output.
    Method:
        This function relies on a "running gaussian average" [1] to compute and
        update what is considered to be the background. Foreground objects are
        detected based on their difference with this background. Resulting 
        foreground images are saved.
    
    References:
        [1] https://en.wikipedia.org/wiki/Background_subtraction#Running_Gaussian_average 
    
    Written by P. DERIAN, 2016.
    """

    def equalize_color(img):
        """Apply  histogram equalization to each channel of the color image.
        
        Note: returns float [0;1]-valued image.
        Note: Not used, in the end.
        """
        imgeq = numpy.zeros_like(img, dtype='float')
        for i in xrange(img.shape[2]):
            imgeq[:,:,i] = exposure.equalize_hist(img[:,:,i])
        return imgeq

    # pre-compute mean, std for initialization
    if verbose:
        print 'Initializing background...'
    imgstack = []
    for k in range(kmin, kmin+kwin):
        fname = os.path.join(indir, inpattern.format(k))
        img = pyplot.imread(fname) # read image
        imgstack.append(img) # add to the stack
    imgstack = numpy.asarray(imgstack)
    imgmean = numpy.average(imgstack, axis=0)
    imgstd = numpy.std(imgstack, axis=0)
    rga = RunningGaussianAverage(imgmean, 15) #15 is the initial (constant) standard deviation


    # now run the detector
    if verbose:
        print 'Running foreground dectector...'
    for k in range(kmin, kmax):
        fname = os.path.join(indir, inpattern.format(k))
        # read image
        img = pyplot.imread(fname) # read image
        # classification mask: foreground if all channels are flagged as foreground
        isfg = numpy.all(rga.update_classify(img.astype('float'), rho=1./kwin), axis=-1)
        # set bg to 0
        img[numpy.logical_not(isfg)] = 0 
        outfile = os.path.join(DIFF_DIR, DIFF_PATTERN.format(k))
        pyplot.imsave(outfile, img)
        if verbose:
            print '\tsaved {}'.format(outfile)
    if verbose:
        print 'Processing complete.'

if __name__=="__main__":
    backgroundSubtraction(KMIN, KMAX, KWIN)