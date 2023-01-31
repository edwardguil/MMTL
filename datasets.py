import os, json, torch, csv, glob, webvtt
from torch.utils.data import Dataset
from torchvision.io import read_video
from torchvision.transforms import Compose, Lambda, Resize, PILToTensor
from pytorchvideo.data.encoded_video import EncodedVideo
from pytorchvideo.transforms import ApplyTransformToKey, UniformTemporalSubsample
from torchvision.transforms._transforms_video import NormalizeVideo
from PIL import Image

class BobslDataset(Dataset):
    def __init__(self, root, split="train", transform_video=True, 
            width=224, height=224, fps=25, mean=[0, 0, 0], std=[1, 1, 1]):
        self.root = root
        self.split = split
        self.video_list = self.get_video_list(root)
        self.transform_video = transform_video
        self.width = width
        self.height = height
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, idx):
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
        return 2

    def get_video_list(self, path):
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
    
class ElarDataset(Dataset):
    '''
    Custom Dataset for Elar Data.
    '''

    def __init__(self, annotations_file, class_file, root, lexicon_file="None", freetransl=True, transform_video=True,
            width=224, height=224, fps=25, mean=[0, 0, 0], std=[1, 1, 1], warnings=True):
        self.video_list = self.read_elar(annotations_file)
        self.class_map = self.read_class_file(class_file)
        self.lexicon_map = self.read_lexicon(lexicon_file)
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
        return len(self.video_list)

    def __getitem__(self, idx):
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
        video = EncodedVideo.from_path(video_path)
        video_data = video.get_clip(start, end)

        # Transforms
        if self.transform_video:
            if int((end - start) * self.fps) > video_data['video'].shape[1]:
                if self.warnings:
                    print(f"Warning! Sampling clip: '{filename}' with start time: {start} will increase frames. Lower fps.")
            transform =  ApplyTransformToKey(
                key="video",
                transform=Compose(
                    [
                        UniformTemporalSubsample(int((end - start) * self.fps)),
                        Resize((self.width, self.height)),
                        Lambda(lambda x: x/255.0),
                        NormalizeVideo(self.mean, self.std)
                        # Lambda(lambda x: x.permute(1, 0, 2, 3)),
                        # Normalize(self.mean, self.std),
                        # Lambda(lambda x: x.permute(1, 0, 2, 3))
                    ]
                ),
            )

            video_data = transform(video_data)
        
        # # Map Glosses To Ints with Lexicon
        # gloss_targets = [[self.lexicon_map[char.upper()] for char in gloss] for gloss in glosses]
        # temp = []
        # for chars in gloss_targets:
        #     for char in chars:
        #         temp.append(char)
        #     temp.append(self.lexicon_map["_"])
        # gloss_targets = temp
        gloss_targets = [self.class_map[gloss] for gloss in glosses]

        return video_data['video'], freetransl, gloss_targets  

    def num_classes(self):
        return len(self.class_map)

    def read_elar(self, path):
        video_list = []
        with open(path, "r") as file:
            data = json.load(file)
        for key, values in data.items():
            for value in values:
                value.update({"FileName" : key})
                video_list.append(value)
        return video_list

    def read_lexicon(self, path):
        return None
        with open(path, "r") as file:
            data = file.readlines()
        return {k[:-1]: v for v, k in enumerate(data)}
    
    def read_class_file(self, path):
        class_names = [c.strip() for c in open(path)]
        return {name : idx for idx, name in enumerate(class_names)}


class PheonixDataset(Dataset):
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

    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, idx):
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
            print(f"FN: {filename}")
            exit()

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
        return len(self.class_map)

    def get_class_map(self, path):
        if path == None:
            print("No class file specified... generating.")
            return self.generate_class_map()
        else:
            with open(path, "r") as file:
                return json.load(file)

    def verify_files(self):
        for split in ["train", "test", "dev"]:
            annontation_name = f"PHOENIX-2014-T.{split}.corpus.csv"

            with open(os.path.join(self.root, "annotations", "manual", annontation_name), 'r') as file:
                csv_rows = list(csv.reader(file))

            video_list = [row[0].split("|") for row in csv_rows[1:]]
            for set in video_list:
                filename = set[0]

    def generate_class_map(self):
        '''
        Gets the entire pheonix vocab and provides an unique class idx for each.
        Should be used only once, then generated file should be passed
        as parameter to class_file. 
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
        return len(self.video_list)

    def __getitem__(self, idx):
        filename = self.video_list[idx][0]
        gloss = self.video_list[idx][1]
        
        video_path = os.path.join(self.root, "videos", f"{filename}.mp4")
        if not os.path.exists(video_path):
            raise ValueError(f"Video {filename}.mp4 dosen't exist. Remove from {self.split} set.")

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
        return len(self.class_map)

    def get_class_map(self, path):
        if path == None:
            print("No class file specified... generating.")
            return self.generate_class_map(path)
        else:
            with open(path, "r") as file:
                return json.load(file)

    
    def generate_class_map(self, path):
        '''
        Gets the entire wlasl vocab and provides an unique class idx for each.
        Should be used only once, then generated file should be passed
        as parameter to class_file. 
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





def elar_collate_fn(batch, device):
    '''
    Should be passed to dataloader. 
    Additional parameter can be passed on call with:
    collate_fn=partial(elar_collate_fn, device=device)
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
    data, targets = zip(*batch)
    data = [ torch.Tensor(t).permute(1, 0, 2, 3).to(device) for t in data ]
    data = torch.nn.utils.rnn.pad_sequence(data, batch_first=True).permute(0, 2, 1, 3, 4)
    targets = torch.tensor(targets).long().to(device)

    return data, targets