from skimage.exposure import rescale_intensity
from tifffile import TiffFile
import numpy as np

def correct_path(path):
    if path[-1] != "/":
        path = path + "/"
    return path

def tif_reader_5D(path_to_file):
    """
    Parameters
    ----------
    path_to_file : str
        The path to the tif file.

    Returns
    -------
    hyperstack:
        5D numpy array with shape (t, z, c, x, y)
    metadata:
        Dict containing imagej metadata and xy and z spacings (inverse of resolution)

    """
    with TiffFile(path_to_file) as tif:
        hyperstack = tif.asarray()
        imagej_metadata = tif.imagej_metadata
        tags = tif.pages[0].tags

        try:
            frames = imagej_metadata["frames"]
        except:
            frames = 1

        try:
            slices = imagej_metadata["slices"]
        except:
            slices = 1

        try:
            channels = imagej_metadata["channels"]
        except:
            channels = 1

        try:
            hyperstack = np.reshape(
                hyperstack, (frames, slices, channels, *hyperstack.shape[-2:])
            )
        except:
            print(
                "WARNING: Could not interpret metadata to reshape hyperstack. Making up dimensions"
            )
            print("         raw array with shape", hyperstack.shape)
            if len(hyperstack.shape) == 2:
                hyperstack = np.reshape(hyperstack, (1, 1, 1, *hyperstack.shape[-2:]))
            elif len(hyperstack.shape) == 3:
                hyperstack = np.reshape(
                    hyperstack, (1, hyperstack.shape[0], 1, *hyperstack.shape[-2:])
                )
            elif len(hyperstack.shape) == 4:
                hyperstack = np.reshape(
                    hyperstack,
                    (
                        hyperstack.shape[0],
                        hyperstack.shape[1],
                        1,
                        *hyperstack.shape[-2:],
                    ),
                )
            print("         returning array with shape", hyperstack.shape)

        # parse X, Y resolution
        try:
            npix, unit = tags["XResolution"].value
            xres = unit / npix
        except:
            xres = 1

        try:
            npix, unit = tags["YResolution"].value
            yres = unit / npix
        except:
            yres = 1

        try:
            res_unit = tags["ResolutionUnit"].value
        except:
            res_unit = 1

        try:
            zres = imagej_metadata["spacing"]
        except:
            zres = 1

        if xres == yres:
            xyres = xres
        else:
            xyres = np.mean([xres, yres])

    if imagej_metadata is None:
        imagej_metadata = {}
    imagej_metadata["XYresolution"] = xyres
    imagej_metadata["Zresolution"] = zres
    imagej_metadata["ResolutionUnit"] = res_unit
    return hyperstack, imagej_metadata


def read_split_times(
    path_data, times, name_format="{}", channels=None
):
    IMGS = []
    extension = None
    if '.' in name_format:
        extension = name_format.split('.')[-1]

    if extension in path_data:
        path_to_file = path_data
        IMGS, metadata = tif_reader_5D(path_to_file)
        if channels is None:
            channels = [i for i in range(IMGS.shape[2])]
        IMGS = IMGS[:, :, channels, :, :]
        times_ids = np.array(times)
        # IMGS = IMGS[times_ids].astype("uint8")
        IMGS = rescale_intensity(IMGS[times_ids], out_range='uint8')
    else:
        for t in times:
            path_to_file = correct_path(path_data) + name_format.format(t)

            if extension in ["tif","tiff"]:
                IMG, metadata = tif_reader_5D(path_to_file)
                if channels is None:
                    channels = [i for i in range(IMG.shape[2])]
                # IMGS.append(IMG[0].astype("uint8"))
                IMG = IMG[:, :, channels, :, :]
                IMGS.append(rescale_intensity(IMG[0], out_range='uint8'))
                del IMG
            elif extension == "npy":
                IMG = np.load(path_to_file)
                IMGS.append(IMG.astype("uint16"))
    
    if extension == "tif":
        return np.array(IMGS), metadata
    elif extension == "npy":
        return np.array(IMGS)


def read_split_vectors(
    path_data, times, mask=None, name_format="{}{}"
):
    
    path_to_file = correct_path(path_data) + name_format.format(times[0], times[1])
    vecs = np.load(path_to_file)
    Vectors = []
    Magnitudes = []

    idmax = 0
    for tid, t in enumerate(times[:-1]):
        path_to_file = correct_path(path_data) + name_format.format(times[tid], times[tid+1])
        vecs = np.load(path_to_file)
        if mask is not None:        
            keep = mask[tid, vecs[:, 0, 0].astype(int), vecs[:, 0, 1].astype(int), vecs[:, 0, 2].astype(int)]
            vecs = vecs[keep,:,:]
        nvecs = vecs.shape[0]

        Vecs = np.zeros((nvecs*2, 5))
        Magnitude = np.zeros((nvecs*2))

        #Ids
        Vecs[:nvecs,0] = range(idmax, idmax+nvecs)
        Vecs[nvecs:2*nvecs,0] = range(idmax, idmax+nvecs)
        #Time
        Vecs[:nvecs,1] = tid
        Vecs[nvecs:2*nvecs,1] = tid+1
        #Pos
        Vecs[:nvecs,2:] = vecs[:,0,:]
        Vecs[nvecs:2*nvecs,2:] = vecs[:,0,:]+vecs[:,1,:]

        #Magnitude
        # Magnitude[:nvecs] = np.log1p(np.sum(vecs[:,1,2:]**2,axis=1).flatten())
        # Magnitude[nvecs:2*nvecs] = np.log1p(np.sum(vecs[:,1,2:]**2,axis=1).flatten())
        Magnitude[:nvecs] = np.sum(vecs[:,1,2:]**2,axis=1).flatten()
        Magnitude[nvecs:2*nvecs] = np.sum(vecs[:,1,2:]**2,axis=1).flatten()
        
        idmax += nvecs
        Vectors.append(Vecs)
        Magnitudes.append(Magnitude)
        
    return np.array(np.vstack(Vectors)), np.concatenate(Magnitudes)
