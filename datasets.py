import os, json, torch, csv, glob, webvtt
from torch.utils.data import Dataset
from torchvision.io import read_video
from torchvision.transforms import Compose, Lambda, Resize, PILToTensor
from pytorchvideo.data.encoded_video import EncodedVideo
from pytorchvideo.transforms import ApplyTransformToKey, UniformTemporalSubsample
from torchvision.transforms._transforms_video import NormalizeVideo
from PIL import Image

class ElarDataset(Dataset):
    """
    PyTorch Dataset to load videos and annotations from the ELAR corpus.

    Args:
        annotations_file (str): Path to the JSON file containing annotations. 
            See github for format.
        class_file (str): Path to the file containing class names.
        root (str): Root directory of the dataset.
        freetransl (bool): Whether to use the free translation provided in annotations.
        transform_video (bool): Whether to apply transformations to video frames.
        width (int): Width of the output video frame.
        height (int): Height of the output video frame.
        fps (int): Frames per second to sample the video.
        mean (list): List of mean values for normalization of the video.
        std (list): List of standard deviation values for normalization of the video.
        warnings (bool): Whether to show warnings during data loading.

    Methods:
        __len__(self): Returns the number of videos in the dataset.
        __getitem__(self, idx): Returns the video data, free translation, and gloss targets for a given index.
        num_classes(self): Returns the number of unique gloss targets in the dataset.
        get_class_names(self): Returns a list of the unique gloss targets in the dataset.
        read_elar(self, path): Reads the annotations from a JSON file and returns a list of videos with associated metadata.
        read_class_file(self, path): Reads the class names from a file and returns a list.
        convert_idx_str(self, glosses): Converts a list of gloss targets to a single string of indices.
    """

    def __init__(self, annotations_file, class_file, root, freetransl=True, transform_video=True,
            width=224, height=224, fps=25, mean=[0, 0, 0], std=[1, 1, 1], warnings=True):
        self.video_list = self.read_elar(annotations_file)
        self.class_map = self.read_class_file(class_file)
        self.root = root
        self.freetransl = freetransl
        self.transform_video = transform_video
        self.width = width
        self.height = height
        self.fps = fps
        self.mean = mean
        self.std = std
        self.warnings = warnings

    def __len__(self):
        """
        Returns the number of clips in the dataset.
        
        Returns:
            int: Number of clips in the dataset.
        """
        return len(self.video_list)

    def __getitem__(self, idx):
        """
        Returns the video data, free translation, and gloss targets for a given index.
        
        Args:
            idx (int): Index of the video in the dataset to retrieve.
        
        Returns:
            tuple: A tuple containing video data, free translation, and gloss targets for the given index.
        """
        filename = self.video_list[idx]["FileName"]
        start = self.video_list[idx]['Start'] / 1000 # Convert ms to s
        end = self.video_list[idx]['End'] / 1000
        freetransl = self.video_list[idx]['FreeTransl']
        glosses = self.video_list[idx]['Glosses']
        video_path = None

        for file in os.listdir(os.path.join(self.root, filename)):
            if file.find(".mp4") != -1:
                video_path = os.path.join(self.root, filename, file)
                break

        if video_path == None:
            raise ValueError(f"{os.path.join(self.root, filename)} dosen't contain .mp4 file.")

        # Extract Video Data
        video_data = {"video" : read_video(video_path, start_pts=start, end_pts=end, pts_unit="sec", output_format="TCHW")[0].permute(1, 0, 2, 3)}

        # Transforms
        if self.transform_video:
            if int((end - start) * self.fps) > video_data['video'].shape[1]:
                if self.warnings:
                    print(f"Warning! Sampling clip: '{filename}' with start time: {start} will increase frames. Either: Verify clip or continue.")
            transform =  ApplyTransformToKey(
                key="video",
                transform=Compose(
                    [
                        UniformTemporalSubsample(int((end - start) * self.fps)),
                        Resize((self.width, self.height)),
                        Lambda(lambda x: x/255.0),
                        NormalizeVideo(self.mean, self.std)
                    ]
                ),
            )

            video_data = transform(video_data)
        
        try:
            gloss_targets = []
            for gloss in glosses:
                if gloss.strip() != "":
                    gloss_targets.append(self.class_map.index(gloss.strip().upper()))
        except Exception as e:
            raise ValueError(f"Gloss Dosen't Exist. Free: {freetransl}. Gloss {gloss}. Verify integrity of class and annotation files.")

        return video_data['video'], freetransl, gloss_targets  

    def num_classes(self):
        """
        Returns the number of unique gloss targets in the dataset.
        
        Returns:
            int: Number of unique gloss targets in the dataset.
        """
        return len(self.class_map)

    def get_class_names(self):
        """
        Returns a list of the unique gloss targets in the dataset.
        
        Returns:
            list: A list of the unique gloss targets in the dataset.
        """
        return list(self.class_map.keys())

    def read_elar(self, path):
        """
        Reads the annotations from a JSON file and returns a list of videos with associated metadata.
        The format should be as follows:
            {
                "Folder Name": [
                {
                    "FreeTransl": "A sentence level translation.",
                    "Start": (int) start time of clip in video,
                    "End": (int) end time of clip in video,
                    "Glosses": [
                        "GLOSS1",
                        "GLOSS2"
                    ],
                    "Idx": (int) global index of this clip ,
                    "FileName": "name of the video contained in this folder"
                },
            .. 
            }

        Args:
            path (str): Path to the JSON file containing annotations.
        
        Returns:
            list: A list of videos with associated metadata.
        """
        video_list = []
        with open(path, "r") as file:
            data = json.load(file)
        for key, values in data.items():
            for value in values:
                value.update({"FileName" : key})
                video_list.append(value)
        return video_list
    
    def read_class_file(self, path):
        """
        Reads the class names (glosses) from a file and returns a list. 
        Each line has a single gloss.
        
        Args:
            path (str): Path to the file containing class names.
        
        Returns:
            list: A list of class names.
        """
        class_names = [c.strip() for c in open(path)]
        self.class_names = [name.upper() for name in class_names]
        return class_names

    def convert_idx_str(self, glosses):
        """
        Converts a list of gloss indices to a single string of glosses.
        
        Args:
            glosses (list): A list of gloss indices to convert.
        
        Returns:
            str: A single string of glosses.
        """
        str = ""
        for gloss in glosses:
            str += self.class_map[gloss]
            str += " "
        return str

class BobslDataset(Dataset):
    """PyTorch Dataset for the BOBSL dataset. 

    Args:
        root (str): The root directory of the BOBSL dataset.
        split (str, optional): The split to use. Must be one of 'train', 'test', or 'val'. 
        transform_video (bool, optional): Whether or not to apply video transforms.
        width (int, optional): The width of the transformed video.
        height (int, optional): The height of the transformed video.
        fps (int, optional): The frames per second of the video.
        mean (list, optional): The mean values for normalization of the video.
        std (list, optional): The standard deviation values for normalization of the video.

    Methods:
        __len__(): Returns the length of the dataset.
        __getitem__(idx): Returns the item at the given index.
        num_class(): Returns the number of classes in the dataset.
        get_video_list(path): Helper method to get a list of videos for the dataset.

    """
    def __init__(self, root, split="train", transform_video=True, 
            width=224, height=224, fps=25, mean=[0, 0, 0], std=[1, 1, 1]):
        """Initializes the BOBSL dataset.

        Args:
            root (str): The root directory of the BOBSL dataset.
            split (str, optional): The split to use. Must be one of 'train', 'test', or 'val'.
            transform_video (bool, optional): Whether or not to apply video transforms.
            width (int, optional): The width of the transformed video.
            height (int, optional): The height of the transformed video.
            fps (int, optional): The frames per second of the video.
            mean (list, optional): The mean values for normalization of the video.
            std (list, optional): The standard deviation values for normalization of the video.
        """
        self.root = root
        self.split = split
        self.video_list = self.get_video_list(root)
        self.transform_video = transform_video
        self.width = width
        self.height = height
        self.mean = mean
        self.std = std
        self.fps = fps

    def __len__(self):
        """Returns the length of the dataset.

        Returns:
            int: The length of the dataset.
        """
        return len(self.video_list)

    def __getitem__(self, idx):
        """Returns the item at the given index.

        Args:
            idx (int): The index of the item to return.

        Returns:
            tuple: A tuple containing the video data, the sentence for the video, and a list of targets.
        """
        filename, start, end, sentence = self.video_list[idx]       

        video_path = os.path.join(self.root, "videos", f"{filename}.mp4")
        if not os.path.exists(video_path):
            raise ValueError(f"{video_path} dosen't exist.")
        
        video_data = {"video" : read_video(video_path, start_pts=start, end_pts=end, pts_unit="sec", output_format="TCHW")[0].permute(1, 0, 2, 3)}
        
        if self.transform_video:
            transform =  ApplyTransformToKey(
                key="video",
                transform=Compose(
                    [
                        Resize((self.width, self.height)),
                        Lambda(lambda x: x/255.0),
                        NormalizeVideo(self.mean, self.std)
                    ]
                ),
            )

            video_data = transform(video_data)

        return video_data["video"], sentence, [0]

    def num_class(self):
        """Disregard. Implemented for continuity. """
        return 2

    def get_video_list(self, path):
        """Helper method taht extracts the list of clips and their captions 
        from all the vtt files.

        Args:
            path (str): The path to the dataset.

        Returns:
            list: A list of videos for the dataset.
        """
        # Helper for converting vtt time into seconds
        to_seconds = lambda x: (lambda y: int(y[0]) * 3600 + int(y[1]) * 60 + float(y[2]))(x.split(":"))
        if self.split not in ["train", "test", "val"]:
            raise ValueError("Split argument must be 'train', 'test' or 'val'.")

        with open(os.path.join(path, "subset2episode.json"), "r") as file:
            filename_set = json.load(file)[self.split]

        video_list = []
        for filename in filename_set:
            for caption in webvtt.read(os.path.join("subtitles", f'{filename}.vtt')):
                video_list.append([filename, to_seconds(caption.start), to_seconds(caption.end), caption.text])
        return video_list

class PheonixDataset(Dataset):
    """
    A PyTorch dataset class for the PHOENIX-2014-T dataset.

    Args:
        root (str): The root directory of the dataset.
        class_file (str, optional): The path to the class file containing a JSON object of class names and indices. 
            If not provided, the class map will be generated and saved to "classes_pheonix.json". 
        split (str, optional): The split of the dataset to use. Must be one of "train", "test", or "val". 
        transform_video (bool, optional): Whether to transform the video data. 
        width (int, optional): The width of the video frames.
        height (int, optional): The height of the video frames. 
        fps (int, optional): The frames per second of the videos.
        mean (list[float], optional): The mean values for normalization. 
        std (list[float], optional): The standard deviation values for normalization.
    
    Methods:
        __len__(): Returns the length of the dataset.
        __getitem__(idx): Returns the item at the given index.
        get_video_list(path): Helper method to get a list of videos for the dataset.
        num_class(): Returns the number of classes in the dataset.
        get_class_names(self): Returns a list of the unique gloss targets in the dataset.
        get_class_map(self): Helper methoding for loading the classes from the class file. 
        generate_class_map(): Gets the entire pheonix vocab and provides an unique class idx for each.

    """
    def __init__(self, root, class_file=None, split="train", transform_video=True, 
            width=224, height=224, fps=25, mean=[0, 0, 0], std=[1, 1, 1]):
        self.root = root
        self.split = split
        self.video_list = self.get_video_list(root)
        self.class_map = self.get_class_map(class_file)
        self.transform_video = transform_video
        self.width = width
        self.height = height
        self.mean = mean
        self.std = std
        self.fps = fps

    def __len__(self):
        """
        Returns the number of samples in the dataset.

        Returns:
            int: The number of sampels in the dataset.
        """
        return len(self.video_list)

    def __getitem__(self, idx):
        """
        Returns the video data, free translation, and gloss targets for a given index.
        
        Args:
            idx (int): Index of the video in the dataset to retrieve.
        
        Returns:
            tuple: A tuple containing video data, free translation, and gloss targets for the given index.
        """
        filename = self.video_list[idx][0]
        glosses = self.video_list[idx][-2].split(" ")
        freetransl = self.video_list[idx][-1]
        
        parent = os.path.join(self.root, "features", "fullFrame-210x260px", self.split, filename)
        image_paths = glob.glob(f"{parent}/*.png")
        image_transform = Compose([PILToTensor()])

        try:
            video = torch.stack([image_transform(Image.open(path)) for path in image_paths]).permute(1, 0, 2, 3)
            video_data = {"video" : video}
        except:
            raise ValueError(f"Folder: {filename} is missing or contains no images. Verify integrity of dataset.")

        if self.transform_video:
            transform =  ApplyTransformToKey(
                key="video",
                transform=Compose(
                    [
                        Resize((self.width, self.height)),
                        Lambda(lambda x: x/255.0),
                        NormalizeVideo(self.mean, self.std)
                    ]
                ),
            )

            video_data = transform(video_data)

        gloss_targets = [self.class_map[gloss] for gloss in glosses]

        return video_data["video"], freetransl, gloss_targets


    def get_video_list(self, path):
        """
        Gets the list of videos in the dataset.

        Args:
            path (str): The path to the root directory of the dataset.

        Returns:
            list: A list of lists, where each inner list contains the filename, gloss sequence, and free translation for a video.
        """
        if self.split in ["train", "test", "val"]:
            self.split =  (lambda x: "dev" if x == "val" else self.split)(self.split)
            annontation_name = f"PHOENIX-2014-T.{self.split}.corpus.csv"
        else:
            raise ValueError("Split argument must be 'train', 'test' or 'val'.")

        with open(os.path.join(path, "annotations", "manual", annontation_name), 'r') as file:
            csv_rows = list(csv.reader(file))

        video_list = [row[0].split("|") for row in csv_rows[1:]]

        return video_list

    def num_classes(self):
        """
        Returns the number of classes in the dataset.

        Returns:
            int: The number of classes in the dataset.
        """
        return len(self.class_map)

    def get_class_names(self):
        """
        Returns a list of the unique gloss targets in the dataset.
        
        Returns:
            list: A list of the unique gloss targets in the dataset.
        """
        return list(self.class_map.keys())

    def get_class_map(self, path):
        """
        Returns a list of the unique gloss targets in the dataset.
        
        Returns:
            list: A list of the unique gloss targets in the dataset.
        """
        if path == None:
            print("No class file specified... generating.")
            return self.generate_class_map()
        else:
            with open(path, "r") as file:
                return json.load(file)

    def generate_class_map(self):
        '''
        Gets the entire pheonix vocab and provides an unique class idx for each.
        Should be used only once, then generated file should be passed
        as parameter to class_file. 

        Returns:
            dict: A dictionary of each unique gloss and its corrosponding id
        '''
        class_map = {}
        for split in ["train", "test", "dev"]:
            annontation_name = f"PHOENIX-2014-T.{split}.corpus.csv"

            with open(os.path.join(self.root, "annotations", "manual", annontation_name), 'r') as file:
                csv_rows = list(csv.reader(file))

            video_list = [row[0].split("|") for row in csv_rows[1:]]
            gloss_list = [row[-2] for row in video_list]
            for gloss_sequence in gloss_list:
                for gloss in gloss_sequence.split():
                    class_map.update({gloss : 0})

        for idx, gloss in enumerate(list(class_map.keys())):
            class_map[gloss] = idx

        json_object = json.dumps(class_map, indent = 4)
        with open("classes_pheonix.json", "w+") as file:
            file.write(json_object)

        return class_map


class WLASLDataset(Dataset):
    """
    A PyTorch dataset class for the WLASL dataset.

    Args:
        root (str): The root directory of the dataset.
        class_file (str, optional): The path to the class file containing a JSON object of class names and indices. 
            If not provided, the class map will be generated and saved to "classes_wlasl.json". 
        split (str, optional): The split of the dataset to use. Must be one of "train", "test", or "val". 
        transform_video (bool, optional): Whether to transform the video data. 
        width (int, optional): The width of the video frames.
        height (int, optional): The height of the video frames. 
        fps (int, optional): The frames per second of the videos.
        mean (list[float], optional): The mean values for normalization. 
        std (list[float], optional): The standard deviation values for normalization.
    
    Methods:
        __len__(): Returns the length of the dataset.
        __getitem__(idx): Returns the item at the given index.
        get_video_list(path): Helper method to get a list of videos for the dataset.
        num_class(): Returns the number of classes in the dataset.
        get_class_map(self): Helper methoding for loading the classes from the class file. 
        generate_class_map(): Gets the entire pheonix vocab and provides an unique class idx for each.

    """

    def __init__(self, root, class_file=None, split="train", transform_video=True, 
            width=224, height=224, fps=25, mean=[0, 0, 0], std=[1, 1, 1]):
        self.root = root
        self.split = split
        self.video_list = self.get_video_list(root)
        self.class_map = self.get_class_map(class_file)
        self.transform_video = transform_video
        self.width = width
        self.height = height
        self.fps = fps
        self.mean = mean
        self.std = std

    def __len__(self):
        """
        Returns the length of the dataset.

        Returns:
            int: The length of the dataset.
        """
        return len(self.video_list)

    def __getitem__(self, idx):
        """
        Returns a tuple pair (video data, label) of the sample at the given index.

        Args:
            idx (int): The index of the item to return.

        Returns:
            tuple: A tuple containing the video data, the corresponding class label.
        """
        filename = self.video_list[idx][0]
        gloss = self.video_list[idx][1]
        
        video_path = os.path.join(self.root, "videos", f"{filename}.mp4")
        if not os.path.exists(video_path):
            raise ValueError(f"Video {filename}.mp4 dosen't exist. Verify dataset integrity or remove from {self.split} set.")

        video = EncodedVideo.from_path(video_path)
        video_data = video.get_clip(0, 1000)

        if self.transform_video:
            transform =  ApplyTransformToKey(
                key="video",
                transform=Compose(
                    [
                        Resize((self.width, self.height)),
                        Lambda(lambda x: x/255.0),
                        NormalizeVideo(self.mean, self.std)
                    ]
                ),
            )

            video_data = transform(video_data)

        return video_data["video"], self.class_map[gloss]

    def get_video_list(self, path):
        """
        Helper method for retrieving the list of videos for the specified split.

        Args:
            path (str): The root directory of the dataset.

        Returns:
            list: A list of video metadata, including the filename, gloss, and fps.
        """
        if self.split not in ["train", "test", "val"]:
            raise ValueError("Split argument must be 'train', 'test' or 'val'.")

        with open(os.path.join(path, "WLASL_v0.3.json"), "r") as file:
            wlasl_info = json.load(file)

        video_list = []
        for class_set in wlasl_info:
            gloss = class_set["gloss"]
            for instance in class_set["instances"]:
                if self.split == instance["split"]:
                    video_list.append([instance["video_id"], gloss, instance["fps"]])

        return video_list

    def num_classes(self):
        """
        Returns the number of classes in the dataset.

        Returns:
            int: The number of classes in the dataset.
        """
        return len(self.class_map)

    def get_class_map(self, path):
        """
        Returns a dictionary mapping class labels to integer indices.

        Args:
            path (str): The path to the class file.

        Returns:
            dict: A dictionary mapping class labels to integer indices.
        """
        if path == None:
            print("No class file specified... generating.")
            return self.generate_class_map()
        else:
            with open(path, "r") as file:
                return json.load(file)

    
    def generate_class_map(self):
        '''
        Gets the entire wlasl vocab and provides an unique class idx for each.
        Should be used only once, then generated file should be passed
        as parameter to class_file. 

        Returns:
            dict: A dictionary mapping class labels to integer indices
        '''
        gloss_list = []
        with open(os.path.join(self.root, "WLASL_v0.3.json"), "r") as file:
            wlasl_info = json.load(file)
        
        for class_set in wlasl_info:
            gloss_list.append(class_set["gloss"])

        class_map = {gloss : idx for idx, gloss in enumerate(gloss_list)}

        json_object = json.dumps(class_map, indent = 4)
        with open("classes_wlasl.json", "w+") as file:
            file.write(json_object)

        return class_map



def batch_mean_and_sd(dataloader):
    '''
    Calculates mean and std of a dataset contained in dataloader. 
    Can be used with Pheonix, ELAR and Bobsl datasets.
    '''
    with torch.no_grad():
        channels_sum, channels_squared_sum, num_batches = 0, 0, 0
        try:
            for step, (data, _, _, _, freetransl) in enumerate(dataloader):
                # Mean over batch, height and width, but not over the channels
                channels_sum += torch.mean(data, dim=[0,2,3,4])
                channels_squared_sum += torch.mean(data**2, dim=[0,2,3,4])
                num_batches += 1

        except Exception as e:
            print(f"Exception Caught: {e}. Step: {step}. Freetransl {freetransl}")
            print(f"Printing info for continuing calculation.")
            print(f"Num Batches: {num_batches}. Channels Sum: {channels_sum}. Channels Squared Sum: {channels_squared_sum}")
            exit()

        mean = channels_sum / num_batches

        std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5

    return mean, std     

def batch_mean_and_sd_std(dataloader):
    '''
    Calculates mean and std of a dataset contained in dataloader. 
    Can be used with Wlasl datasets.
    '''
    with torch.no_grad():
        channels_sum, channels_squared_sum, num_batches = 0, 0, 0
        for step, (data, target) in enumerate(dataloader):
            # Mean over batch, height and width, but not over the channels
            channels_sum += torch.mean(data, dim=[0,2,3,4])
            channels_squared_sum += torch.mean(data**2, dim=[0,2,3,4])
            num_batches += 1
        
        mean = channels_sum / num_batches

        std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5

    return mean, std  

def collate_fn(batch, device):
    '''
    Used for transforming various datasets into a format required for training. 
    Should be used to overide default collate_fn in dataloader.
    '''
    data, freetransl, gloss_targets = zip(*batch)
    data_lengths = torch.ceil(torch.Tensor([ t.shape[1]/4 for t in data])).int().to(device)
    gloss_lengths = torch.Tensor([ len(t) for t in gloss_targets]).int().to(device)

    data = [ torch.Tensor(t).permute(1, 0, 2, 3).to(device) for t in data ]
    data = torch.nn.utils.rnn.pad_sequence(data, batch_first=True).permute(0, 2, 1, 3, 4)
    gloss_targets = [ torch.Tensor(t).int().to(device) for t in gloss_targets ]
    gloss_targets = torch.nn.utils.rnn.pad_sequence(gloss_targets, batch_first=True)

    return data, data_lengths, gloss_targets, gloss_lengths, freetransl

def s3d_collate_fn(batch, device):
    '''
    Used for transforming WLASL dataset into a format required for training. 
    Should be used to overide default collate_fn in dataloader. 
    '''
    data, targets = zip(*batch)
    data = [ torch.Tensor(t).permute(1, 0, 2, 3).to(device) for t in data ]
    data = torch.nn.utils.rnn.pad_sequence(data, batch_first=True).permute(0, 2, 1, 3, 4)
    targets = torch.tensor(targets).long().to(device)

    return data, targets