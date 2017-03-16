###
###
import numpy
import matplotlib.pyplot as pyplot
###

def velocity_map(filename, bgimg=None):
    """
    """
    # load data
    data = numpy.loadtxt(filename)
    frame = data[:,0].astype(int)
    px = data[:,1] # coordinates
    py = data[:,2]
    ux = data[:,3] # displacement
    uy = data[:,4]
    norm = numpy.sqrt(ux**2 + uy**2)
    nframes = numpy.unique(frame).size
    
    
    dpi = 90.
    fig = pyplot.figure(figsize=(1280./dpi, 720./dpi))
    ax = pyplot.gca() 
    ax.set_xlabel('x [px]')
    ax.set_ylabel('y [px]')   
    ax.set_title('Magnitude of estimated particle displacements [pixel] over {} frames ({} vectors)'.format(nframes, frame.size))
    sc = ax.scatter(px, py, 6, c=norm, alpha=1.0, edgecolors='none',
                    cmap='plasma', vmin=6., vmax=16)
    pyplot.colorbar(sc, shrink=0.5, fraction=0.05)
    if bgimg is not None:
        img = pyplot.imread(bgimg)
        pyplot.imshow(img, origin='upper')
    pyplot.subplots_adjust(left=0.06, bottom=0.07, right=1.0, top=0.94)
    fig.savefig('velocity_map.png', dpi=dpi)
    pyplot.show()

if __name__=="__main__":
    velocity_map(
        '/Users/pderian/Documents/Data/SourcesHydro_ifremer/sequences-video-flux-p_rodier/rodier_seq5-MOMARSAT13_MOMARSAT1303_130828142725_15/track/track.txt',
        '/Users/pderian/Documents/Data/SourcesHydro_ifremer/sequences-video-flux-p_rodier/rodier_seq5-MOMARSAT13_MOMARSAT1303_130828142725_15/0300.jpg',
                )