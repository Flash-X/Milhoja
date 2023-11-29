todo::
    * The variable masking in gpu_tf_ener.json should be using [2,2] for both tile_in and tile_out arrays.
      However, because of the way farray and the datapacket generator work right now, we are forced to use
      [1,2] as the variable mask for the tile_in. When reworking the FArray classes, we should update them
      to allow them to use different indexing spaces.
