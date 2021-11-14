import solaris.raster_tile as raster_tile

raster_tiler = raster_tile.RasterTiler(dest_dir='res', 
                                       src_tile_size=(640, 640),
                                       aoi_boundary=box(*test_bounds),
                                       verbose=True)

out_dir = '/content/SN6_20190823162315_20190823162606.tif'
raster_bounds_crs = raster_tiler.tile(out_dir, ver_name='base',
                                      nodata_threshold=0.5)