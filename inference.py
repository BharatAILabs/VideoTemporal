import scripts.Berry_Method as General_SD
import os
import shutil
import scripts.Ebsynth_Processing as ebsynth_process
import scripts.berry_utility as sd_utility
from PIL import Image
import re
import numpy as np
from typing import Union, List

def numpy_array_to_temp_url(img_array: np.ndarray) -> str:
    # create a filename for the temporary file
    filename = 'generatedsquare.png'
    extension_path = os.path.abspath(__file__)
    extension_dir =  os.path.dirname(extension_path)
    extension_folder = os.path.join(extension_dir,"squares")

    if not os.path.exists(extension_folder):
        os.makedirs(extension_folder)
    
    file_path = os.path.join(extension_folder, filename)
    img = Image.fromarray(img_array)
    img.save(file_path, format='PNG')
    return file_path

def preprocess_video(video: str,
                     fps: int = 25,batch_size: int = 5,per_side:int = 2,
                     resolution: int = 1024, batch_run: bool = True,
                     max_frames: int = -1, output_path: str = "batch",
                     border_frames: int = 2,ebsynth_mode: bool = True, split_video: bool = True,
                     split_based_on_cuts: bool = False) -> Union[str, np.ndarray]:

    input_folder_loc = os.path.join(output_path, "input")
    output_folder_loc = os.path.join(output_path, "output")
    if not os.path.exists(input_folder_loc):
        os.makedirs(input_folder_loc)
    if not os.path.exists(output_folder_loc):
        os.makedirs(output_folder_loc)
    
    max_keys = max_frames
    if max_keys < 0:
        max_keys = 10000
    max_frames = (max_frames * (batch_size))
    if max_frames < 1:
        max_frames = 10000
    
    if ebsynth_mode == True:
        if split_video == False:
            border_frames = 0
        if batch_run == False:
            max_frames = per_side * per_side * (batch_size + 1)

        
    if split_video == True:
        border_frames = border_frames * batch_size

        max_frames = (20 * batch_size) - border_frames
        max_total_frames = int((max_keys / 20) * max_frames)
        existing_frames = [] 
           
        if split_based_on_cuts == True:
            existing_frames = sd_utility.split_video_into_numpy_arrays(video,fps,False)
        else:
            data = General_SD.convert_video_to_bytes(video)
            existing_frames = [sd_utility.extract_frames_movpie(data, fps,max_frames=max_total_frames,perform_interpolation=False)]
    

        split_video_paths,transition_data = General_SD.split_videos_into_smaller_videos(max_keys,existing_frames,fps,max_frames,output_path,border_frames,split_based_on_cuts)
        for index,individual_video in enumerate(split_video_paths):
            
            generated_textures = General_SD.generate_squares_to_folder(individual_video,fps=fps,batch_size=batch_size, resolution=resolution,size_size=per_side,max_frames=None, output_folder=os.path.dirname(individual_video),border=0, ebsynth_mode=ebsynth_mode,max_frames_to_save=max_frames)
            input_location = os.path.join(os.path.dirname(os.path.dirname(individual_video)),"input")
            for tex_index,texture in enumerate(generated_textures):
                individual_file_name = os.path.join(input_location,f"{index}and{tex_index}.png")
                General_SD.save_square_texture(texture,individual_file_name)
        transitiondatapath = os.path.join(output_path,"transition_data.txt")
        with open(transitiondatapath, "w") as f:
            f.write(str(transition_data) + "\n")
            f.write(str(border_frames) + "\n")
        main_video_path = os.path.join(output_path,"main_video.mp4")
        sd_utility.copy_video(video,main_video_path)
        return main_video_path
    
    new_video_loc = os.path.join(output_path, f"input_video.mp4")
    shutil.copyfile(video,new_video_loc)
    if ebsynth_mode == True:
        border = 0
        
        image = General_SD.generate_squares_to_folder(video,fps=fps,batch_size=batch_size, resolution=resolution,size_size=per_side,max_frames=max_frames, output_folder=output_path,border=border_frames, ebsynth_mode=True,max_frames_to_save=max_frames)
        return image[0]

    if batch_run == False:
        image = General_SD.generate_square_from_video(video,fps=fps,batch_size=batch_size, resolution=resolution,size_size=per_side )
        processed = numpy_array_to_temp_url(image)
    else:
        image = General_SD.generate_squares_to_folder(video,fps=fps,batch_size=batch_size, resolution=resolution,size_size=per_side,max_frames=max_frames, output_folder=output_path,border=border_frames,ebsynth_mode=False,max_frames_to_save=max_frames)
        processed = image[0]
    return processed

def atoi(text: str) -> Union[int, str]:
    return int(text) if text.isdigit() else text

def natural_keys(text:str) -> Union[List[str], List[int]]:
    return [atoi(c) for c in re.split(r'(\d+)', text)]

def read_images_folder(folder_path: str) -> List[np.ndarray]:
    images = []
    filenames = os.listdir(folder_path)
    
    # Sort filenames based on the order of the numbers in their names
    filenames.sort(key=natural_keys)

    for filename in filenames:
        # Check if file is an image (assumes only image files are in the folder)
        if (filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.jpeg')) and (not re.search(r'-\d', filename)):
            if re.match(r".*(input).*", filename):
                # Open image using Pillow library
            
                img = Image.open(os.path.join(folder_path, filename))
                # Convert image to NumPy array and append to images list
                images.append(np.array(img))
            else:
                print(f"[${filename}] File name must contain \"input\". Skip processing.")
    return images

def post_process_ebsynth(input_folder: str,
                        fps: int = 25,per_side: int = 2,
                        output_resolution:int = 2160,batch_size: int = 5, max_frames: int = 90,
                        border_frames: int = 2):
    
    input_images_folder = os.path.join (input_folder,"output")
    video = os.path.join(input_folder, "input_video.mp4")

    images = read_images_folder(input_images_folder)

    split_mode = os.path.join(input_folder, "keys")
    if os.path.exists(split_mode):
        return ebsynth_process.sort_into_folders(video_path=video,fps=fps,per_side=per_side,batch_size=batch_size,_smol_resolution=output_resolution,square_textures=images,max_frames=max_frames,output_folder=input_folder,border=border_frames)
    else:
        img_folder = os.path.join(input_folder, "output")
        # define a regular expression pattern to match directory names with one or more digits
        pattern = r'^\d+$'

        # get a list of all directories in the specified path
        all_dirs = os.listdir(input_folder)

        # use a list comprehension to filter the directories based on the pattern
        numeric_dirs = sorted([d for d in all_dirs if re.match(pattern, d)], key=lambda x: int(x))
        max_frames = max_frames + border_frames
        for d in numeric_dirs:
            # create a list to store the filenames of the images that match the directory name
            img_names = []
            folder_video = os.path.join(input_folder, d, "input_video.mp4")
            # loop through each image file in the image folder
            for img_file in os.listdir(img_folder):
                # check if the image filename starts with the directory name followed by the word "and" and a sequence of one or more digits, then ends with '.png'
                if re.match(f"^{d}and\d+.*\.png$", img_file):
                    img_names.append(img_file)
            print(f"post processing = {os.path.dirname(folder_video)}")
            square_textures = []
            # loop through each image file name
            for img_name in sorted(img_names, key=lambda x: int(re.search(r'and(\d+)', x).group(1))):
                img = Image.open(os.path.join(input_images_folder, img_name))
                # Convert image to NumPy array and append to images list
                print(f"saving {os.path.join(input_images_folder, img_name)}")
                square_textures.append(np.array(img))

            ebsynth_process.sort_into_folders(video_path=folder_video, fps=fps, per_side=per_side, batch_size=batch_size,
                                    _smol_resolution=output_resolution, square_textures=square_textures,
                                    max_frames=max_frames, output_folder=os.path.dirname(folder_video),
                                    border=border_frames)

def recombine_ebsynth(input_folder,fps,border_frames,batch):
    if os.path.exists(os.path.join(input_folder, "keys")):
        return ebsynth_process.crossfade_folder_of_folders(input_folder,fps=fps,return_generated_video_path=True)
    else:
        generated_videos = []
        pattern = r'^\d+$'

        # get a list of all directories in the specified path
        all_dirs = os.listdir(input_folder)

        # use a list comprehension to filter the directories based on the pattern
        numeric_dirs = sorted([d for d in all_dirs if re.match(pattern, d)], key=lambda x: int(x))

        for d in numeric_dirs:
            folder_loc = os.path.join(input_folder,d)
            # loop through each image file in the image folder
            new_video =  ebsynth_process.crossfade_folder_of_folders(folder_loc,fps=fps)
            #print(f"generated new video at location {new_video}")
            generated_videos.append(new_video)
        
        overlap_data_path = os.path.join(input_folder,"transition_data.txt")
        with open(overlap_data_path, "r") as f:
            merge = str(f.readline().strip())

        overlap_indicies = []
        int_list = eval(merge)
        for num in int_list:
            overlap_indicies.append(int(num))



        output_video = sd_utility.crossfade_videos(video_paths=generated_videos,fps=fps,overlap_indexes= overlap_indicies,num_overlap_frames= border_frames,output_path=os.path.join(input_folder,"output.mp4"))
        return output_video
    return None

def ezsynth_process(style_folder: str, image_folder: str, output_folder: str):

    from ezsynth import Ezsynth
    from glob import glob
    style_paths = list(glob(f'{style_folder}/*'))

    ez = Ezsynth(
        styles=style_paths,
        imgsequence=image_folder,
        edge_method="Classic",
        flow_method="RAFT",
        model="sintel",
        output_folder=output_folder,
    )

    ez.run()  # Run the stylization process
    results = ez.results  # The results are stored in the results variable      
preprocess_video_params = {
    "video": "assets/sample_video.mp4",
    "fps": 25,
    "resolution": 2160
}

# preprocess_video(**preprocess_video_params)

# post_process_ebsynth("./batch/0")

ezsynth_process("/Users/tushargoel/Documents/VideoTemporal/batch/0/keys","/Users/tushargoel/Documents/VideoTemporal/batch/0/frames",
                "/Users/tushargoel/Documents/VideoTemporal/batch/0/output-ezsynth")
