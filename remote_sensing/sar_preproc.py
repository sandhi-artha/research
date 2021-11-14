import solaris.pipesegment as pipesegment
import solaris.image as image
import solaris.sar as sar

import numpy as np
import os


class SarPreproc(pipesegment.PipeSegment):
    def __init__(self,
                 cfg,
                 timestamp='20190823162315_20190823162606',
                 input_dir='expanded',
                 out_dir='results',
                 out_fn='output.tif'
                 ):
        super().__init__()
        out_path = os.path.join(out_dir, out_fn)

        # load polarimetry slc rasters and correct with capella's scale factor
        # quads = []
        # for pol in cfg['pol']:
            # in_fn = os.path.join(input_dir, 'CAPELLA_ARL_SM_SLC_' + pol + '_' + timestamp + '.tif')
            # quads.append(
            #     image.LoadImage(in_fn) * sar.CapellaScaleFactor()
            # )

        quads = [
            image.LoadImage(os.path.join(input_dir, 'CAPELLA_ARL_SM_SLC_'
                                         + pol + '_' + timestamp + '.tif'))
            * sar.CapellaScaleFactor()
            for pol in cfg['pol']]

        self.feeder = (
            np.sum(quads)  # in pipesegment, add operator calls MergeSegment
            * image.MergeToStack()
            * sar.Intensity()
            * sar.Multilook(kernel_size=cfg['ml_size'], method=cfg['ml_filter'])
            * sar.Decibels()
            * sar.Orthorectify(projection=32631, row_res=.5, col_res=.5)
            * image.SaveImage(out_path, return_image=False, no_data_value='nan')
        )