import os
import ffmpy


def convert_tif_to_jpg(tif_path: str, jpg_path: str):
    for filename in os.listdir(tif_path)[475:]:
        output_filename = filename.replace('.tif', '.jpg')
        options = '-q:v 10'

        ff = ffmpy.FFmpeg(
            inputs={tif_path + filename: None},
            outputs={jpg_path + output_filename: options}
        )
        ff.run()
