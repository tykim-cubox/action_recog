from dataset_classification import VideoClsDataset

anno_path = '/home/aiteam/tykim/video_recog/dataset/mix_dataset/anno/train.txt'
data_path = '/home/aiteam/tykim/video_recog/dataset/mix_dataset/train'
ds = VideoClsDataset(anno_path, data_path)

