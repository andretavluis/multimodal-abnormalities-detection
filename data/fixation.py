import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib  import image
import sys


def gaussian(x, sx, y=None, sy=None):
    """Returns an array of numpy arrays (a matrix) containing values between
    1 and 0 in a 2D Gaussian distribution

    arguments
    x		-- width in pixels
    sx		-- width standard deviation

    keyword argments
    y		-- height in pixels (default = x)
    sy		-- height standard deviation (default = sx)
    """

    # square Gaussian if only x values are passed
    if y == None:
        y = x
    if sy == None:
        sy = sx
    # centers
    xo = x/2
    yo = y/2
    # matrix of zeros
    M = np.zeros([y, x], dtype=float)
    # gaussian matrix
    for i in range(x):
        for j in range(y):
            M[j, i] = np.exp(-1.0 * (((float(i)-xo)**2/(2*sx*sx)
                                    ) + ((float(j)-yo)**2/(2*sy*sy))))

    return M

def get_heatmap(fix, dispsize, pupil=False, alpha=0.5, savefilename=None):
    """Draws a heatmap of the provided fixations, optionally drawn over an
    image, and optionally allocating more weight to fixations with a higher
    duration.

    arguments

    fixations		-	a list of fixation ending events from a single trial,
                                    as produced by edfreader.read_edf, e.g.
                                    edfdata[trialnr]['events']['Efix']
    dispsize		-	tuple or list indicating the size of the display,
                                    e.g. (1024,768)

    keyword arguments

    imagefile		-	full path to an image file over which the heatmap
                                    is to be laid, or None for no image; NOTE: the image
                                    may be smaller than the display size, the function
                                    assumes that the image was presented at the centre of
                                    the display (default = None)
    durationweight	-	Boolean indicating whether the fixation duration is
                                    to be taken into account as a weight for the heatmap
                                    intensity; longer duration = hotter (default = True)
    alpha		-	float between 0 and 1, indicating the transparancy of
                                    the heatmap, where 0 is completely transparant and 1
                                    is completely untransparant (default = 0.5)
    savefilename	-	full path to the file in which the heatmap should be
                                    saved, or None to not save the file (default = None)

    returns

    fig			-	a matplotlib.pyplot Figure instance, containing the
                                    heatmap
    """

    # HEATMAP
    # Gaussian
    gwh = 200
    gsdwh = gwh/6
    gaus = gaussian(gwh, gsdwh)
    # matrix of zeroes
    strt = int(gwh/2)
    heatmapsize = int(dispsize[1] + 2*strt), int(dispsize[0] + 2*strt)
    heatmap = np.zeros(heatmapsize, dtype=float)
    # create heatmap
    for i in range(0, len(fix['dur'])):
        # get x and y coordinates
        # x and y - indexes of heatmap array. must be integers
        x = strt + int(fix['x'][i]) - int(gwh/2)
        y = strt + int(fix['y'][i]) - int(gwh/2)
        # correct Gaussian size if either coordinate falls outside of
        # display boundaries
        if (not 0 < x < dispsize[0]) or (not 0 < y < dispsize[1]):
            hadj = [0, gwh]
            vadj = [0, gwh]
            if 0 > x:
                hadj[0] = abs(x)
                x = 0
            elif dispsize[0] < x:
                hadj[1] = gwh - int(x-dispsize[0])
            if 0 > y:
                vadj[0] = abs(y)
                y = 0
            elif dispsize[1] < y:
                vadj[1] = gwh - int(y-dispsize[1])
            # add adjusted Gaussian to the current heatmap
            try:
                if pupil:
                    heatmap[y:y+vadj[1],x:x+hadj[1]] += gaus[vadj[0]:vadj[1],hadj[0]:hadj[1]] * fix['pupil'][i]
                else:
                    heatmap[y:y+vadj[1],x:x+hadj[1]] += gaus[vadj[0]:vadj[1],hadj[0]:hadj[1]] * fix['dur'][i]
            except:
                # fixation was probably outside of display
                pass
        else:
            # add Gaussian to the current heatmap
            if pupil:
                heatmap[int(y):int(y+gwh),int(x):int(x+gwh)] += gaus * fix['pupil'][i]
            else:
                heatmap[int(y):int(y+gwh),int(x):int(x+gwh)] += gaus * fix['dur'][i]

    # resize heatmap
    heatmap = heatmap[strt:dispsize[1]+strt, strt:dispsize[0]+strt]
    # remove zeros
    lowbound = np.mean(heatmap[heatmap > 0])
    # heatmap[heatmap < lowbound] = np.NaN
    heatmap[heatmap < lowbound] = 0

    return heatmap


def word_said_in_fix(fixation_df, timestamps_transcription_df):

    for index1 in fixation_df.index:

        words_said = ''
        count_it = 0

        for index2 in timestamps_transcription_df.index:
            # considered words said 1 second before or after the word is said
            tolerance = 1

            start_fixation = fixation_df.at[index1, 'timestamp_start_fixation']
            start_word = timestamps_transcription_df.at[index2, 'timestamp_start_word']-tolerance

            end_fixation = fixation_df.at[index1, 'timestamp_end_fixation']
            end_word = timestamps_transcription_df.at[index2, 'timestamp_end_word']+tolerance 

            if start_fixation >= start_word and end_fixation <= end_word and count_it>0:
                words_said = words_said + ' ' + timestamps_transcription_df.at[index2, 'word']

            if start_fixation >= start_word and end_fixation <= end_word and count_it==0:
                words_said = words_said + timestamps_transcription_df.at[index2, 'word']
                count_it+=1

            if words_said == '':
                fixation_df.at[index1,'word'] = 'silence'

            else:
                fixation_df.at[index1,'word'] = words_said

    return fixation_df


def get_fixations_dict_from_fixation_df(fixation_df, timestamps_transcription_df = None, first_third = False, second_third=False, rad_silence=False, rad_speaking=False):

    if first_third:
        subindex = int(len(fixation_df) / 3)
        fixation_df['duration']=fixation_df[:subindex]['timestamp_end_fixation'] - fixation_df[:subindex]['timestamp_start_fixation'] 


        return { 
        'x': np.array(fixation_df[:subindex]['x_position']),
        'y': np.array(fixation_df[:subindex]['y_position'] ),
        'dur': np.array(fixation_df[:subindex]['duration']),
        # 'dx': np.array(fixation_df['dx'][1:]),
        # 'dy': np.array(fixation_df['dy'][1:]),
        'pupil': np.array(fixation_df[:subindex]['pupil_area_normalized'])
        }

    if second_third:
        one_third = int(len(fixation_df) / 3)
        two_thirds = int(len(fixation_df) / 3)
        fixation_df['duration']=fixation_df[one_third:two_thirds]['timestamp_end_fixation'] - fixation_df[one_third:two_thirds]['timestamp_start_fixation'] 


        return { 
        'x': np.array(fixation_df[one_third:two_thirds]['x_position']),
        'y': np.array(fixation_df[one_third:two_thirds]['y_position'] ),
        'dur': np.array(fixation_df[one_third:two_thirds]['duration']),
        # 'dx': np.array(fixation_df['dx'][1:]),
        # 'dy': np.array(fixation_df['dy'][1:]),
        'pupil': np.array(fixation_df[one_third:two_thirds]['pupil_area_normalized'])
        }

    if rad_silence:

        fixation_df = word_said_in_fix(fixation_df, timestamps_transcription_df)
        fixation_df = fixation_df.loc[fixation_df['word']=='silence']
        fixation_df['duration']=fixation_df['timestamp_end_fixation'] - fixation_df['timestamp_start_fixation'] 

        return { 
        'x': np.array(fixation_df['x_position']),
        'y': np.array(fixation_df['y_position'] ),
        'dur': np.array(fixation_df['duration']),
        'pupil': np.array(fixation_df['pupil_area_normalized'])
        }

    if rad_speaking:

        fixation_df = word_said_in_fix(fixation_df, timestamps_transcription_df)
        fixation_df = fixation_df.loc[fixation_df['word']!='silence']
        fixation_df['duration']=fixation_df['timestamp_end_fixation'] - fixation_df['timestamp_start_fixation'] 

        return { 
        'x': np.array(fixation_df['x_position']),
        'y': np.array(fixation_df['y_position'] ),
        'dur': np.array(fixation_df['duration']),
        'pupil': np.array(fixation_df['pupil_area_normalized'])
        }

    else:

        fixation_df['duration']=fixation_df['timestamp_end_fixation'] - fixation_df['timestamp_start_fixation'] 

        return { 
            'x': np.array(fixation_df['x_position']),
            'y': np.array(fixation_df['y_position']),
            'dur': np.array(fixation_df['duration']),
            'pupil': np.array(fixation_df['pupil_area_normalized'])
        }


