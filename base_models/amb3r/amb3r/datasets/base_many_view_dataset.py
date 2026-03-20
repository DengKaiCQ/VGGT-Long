# References:
#   https://github.com/HengyiWang/amb3r/blob/main/amb3r/datasets/base_many_view_dataset.py

from dust3r.datasets.base.base_stereo_view_dataset import BaseStereoViewDataset



class BaseManyViewDataset(BaseStereoViewDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def sample_frames(self, img_idxs, rng, d_n='none'):

        img_idxs = list(img_idxs)

        if len(img_idxs) <= self.num_frames:
            # original one + reverse one
            img_idxs = img_idxs + img_idxs[::-1]
        
        while len(img_idxs) < self.num_frames:
            img_idxs = img_idxs + img_idxs

        num_frames = self.num_frames
        #  check whether the train_raio is set
        if not hasattr(self, 'train_ratio'):
            self.train_ratio = 0.5
            # Set a warning message if train_ratio is not set
            print("Warning: train_ratio is not set, using default value of 0.5")
        
        thresh = int(self.min_thresh + self.train_ratio * (self.max_thresh - self.min_thresh))
                
        img_indices = list(range(len(img_idxs)))
        
        selected_indices = []
        
        initial_valid_range = max(len(img_indices)//num_frames, len(img_indices) - thresh * (num_frames - 1))

        try:
            current_index = rng.choice(img_indices[:initial_valid_range])
        except:
            print(f"Warning: Not enough {img_idxs} frames in video {d_n} to sample {self.num_frames} frames")
            raise ValueError(f"Warning: Not enough {img_idxs} frames in video {d_n} to sample {self.num_frames} frames")

        selected_indices.append(current_index)
        
        while len(selected_indices) < num_frames:
            next_min_index = current_index + 1
            next_max_index = min(current_index + thresh, len(img_indices) - (num_frames - len(selected_indices)))
            possible_indices = [i for i in range(next_min_index, next_max_index + 1) if i not in selected_indices]
        
            if not possible_indices:
                break
            
            current_index = rng.choice(possible_indices)
            selected_indices.append(current_index)
        
        if len(selected_indices) < num_frames:
            return self.sample_frames(img_idxs, rng)

        selected_img_ids = [img_idxs[i] for i in selected_indices]
        
        if rng.choice([True, False]):
            selected_img_ids.reverse()
        
        return selected_img_ids
    

    def sample_frame_idx(self, img_idxs, rng, full_video=False, d_n='none'):
        if not full_video:
            img_idxs = self.sample_frames(img_idxs, rng, d_n)
        else:
            img_idxs = img_idxs[::self.kf_every]
        
        return img_idxs


    