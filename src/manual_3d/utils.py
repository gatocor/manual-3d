from skimage.exposure import rescale_intensity
from tifffile import TiffFile
import numpy as np
import os
import re

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

def read_split_vectors_tuple(
    path_data, times, mask=None, name_format="{}{}"
):
    
    path_to_file = correct_path(path_data) + name_format.format(times[0], times[1])
    vecs = np.load(path_to_file)
    Vectors = []
    Vectors2 = []

    idmax = 0
    for tid, t in enumerate(times[:-1]):
        path_to_file = correct_path(path_data) + name_format.format(times[tid], times[tid+1])
        vecs = np.load(path_to_file)
        if mask is not None:        
            keep = mask[tid, vecs[:, 0, 0].astype(int), vecs[:, 0, 1].astype(int), vecs[:, 0, 2].astype(int)]
            vecs = vecs[keep,:,:]
        nvecs = vecs.shape[0]

        Vecs = np.zeros((nvecs, 5))
        Vecs2 = np.zeros((nvecs, 5))

        #Ids
        Vecs[:nvecs,0] = range(idmax, idmax+nvecs)
        Vecs2[:nvecs,0] = range(idmax, idmax+nvecs)
        #Time
        Vecs[:nvecs,1] = tid
        Vecs2[:nvecs,1] = tid+1
        #Pos
        Vecs[:nvecs,2:] = vecs[:,0,:]
        Vecs2[:nvecs,2:] = vecs[:,0,:]+vecs[:,1,:]

        
        idmax += nvecs
        Vectors.append(Vecs)
        Vectors2.append(Vecs2)
        
    # return np.array(np.vstack(Vectors)), np.array(np.vstack(Vectors2))
    return np.array(np.vstack(Vectors))[:,1:], np.array(np.vstack(Vectors2))[:,1:]

def count_files_in_folder(folder_path):
    """Count the number of files in the selected folder and set the start and end points."""
    try:
        file_list = os.listdir(folder_path)
        return len(file_list)
    except Exception as e:
        print(f"Error counting files: {e}")

def detect_file_pattern(folder_path):
    """Detect the file naming pattern, start and end points, and suggest it as the format."""
    try:
        file_list = os.listdir(folder_path)
        file_list = [f for f in file_list if os.path.isfile(os.path.join(folder_path, f))]
        file_list.sort()  # Ensure the files are sorted correctly
        if len(file_list) < 2:
            return  # We need at least two files to detect a pattern
        # Prepare to capture numeric patterns and invalid files
        num_pattern = re.compile(r'\d+')
        valid_files = []
        numeric_values = []
        format_str = None
        for file_name in file_list:
            match = num_pattern.search(file_name)
            if match:
                valid_files.append(file_name)
                numeric_values.append(int(match.group(0)))
            else:
                print(f"File {file_name} does not follow the numeric pattern and will be skipped.")
        if len(valid_files) < 2:
            print("Not enough valid files to detect a pattern.")
            return
        # Compare first and last valid filenames
        file_1, file_2 = valid_files[0], valid_files[-1]
        format_str_parts = []
        i = 0
        while i < len(file_1):
            if i < len(file_2) and file_1[i] == file_2[i]:
                format_str_parts.append(file_1[i])
            elif num_pattern.match(file_1[i:]):
                num_match_1 = num_pattern.match(file_1[i:])
                num_match_2 = num_pattern.match(file_2[i:])
                if num_match_1 and num_match_2:
                    length_1 = len(num_match_1.group(0))
                    length_2 = len(num_match_2.group(0))
                    if length_1 == length_2:
                        format_str_parts.append(f"{{:0{length_1}d}}")
                        i += length_1 - 1  # Skip the numeric section
                    else:
                        format_str_parts.append(file_1[i])
            else:
                format_str_parts.append(file_1[i])
            i += 1
        # Join format string and update the format input
        format_str = ''.join(format_str_parts)

        return format_str, numeric_values
    except Exception as e:
        print(f"Error detecting file pattern: {e}")

def tracks_to_matrix(tracks, scale=np.array((2,0.347,0.347))):

    x = len(np.unique(tracks[:,0]))
    y = sum(tracks[:,0]==0)
    z = 3

    m = np.zeros([x,y,z])
    ids = np.round(tracks[:,0]).astype(int)
    times = np.round(tracks[:,1]).astype(int)
    m[ids,times,0] = tracks[:,2]
    m[ids,times,1] = tracks[:,3]
    m[ids,times,2] = tracks[:,4]
    m *= scale.reshape(1,1,-1)

    return m, ids, times

def displacement_total(tracks, scale=np.array((2,0.347,0.347))):

    trackM, ids, times = tracks_to_matrix(tracks, scale=scale)
    
    v = np.sqrt(np.power((trackM[:,0,:]-trackM[:,-1,:]),2).sum(axis=1))
    v = np.zeros(trackM.shape[:-1]) + v.reshape(-1,1)
    
    return v[ids, times]

def path_total(tracks, scale=np.array((2,0.347,0.347))):

    trackM, ids, times = tracks_to_matrix(tracks, scale = scale)

    v = np.sqrt(np.power(np.diff(trackM,axis=1),2).sum(axis=2)).sum(axis=1)
    v = np.zeros(trackM.shape[:-1]) + v.reshape(-1,1)
    
    return v[ids, times]

def displacement_cumulative(tracks, scale=np.array((2,0.347,0.347))):

    trackM, ids, times = tracks_to_matrix(tracks, scale = scale)

    v = np.sqrt(np.power((trackM[:,0,:].reshape(trackM.shape[0],1,trackM.shape[2])-trackM),2).sum(axis=2))
    
    return v[ids, times]

def path_cumulative(tracks, scale=np.array((2,0.347,0.347))):

    trackM, ids, times = tracks_to_matrix(tracks, scale = scale)

    v = np.sqrt(np.power(np.diff(trackM,axis=1),2).sum(axis=2)).cumsum(axis=1)
    v = np.hstack([np.zeros([v.shape[0],1]),v])
    
    return v[ids, times]