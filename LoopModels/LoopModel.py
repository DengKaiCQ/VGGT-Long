import torch
import argparse
from PIL import Image
import torchvision.transforms as T
from tqdm import tqdm
from pathlib import Path

from LoopModels.vpr_model import VPRModel

class LoopDetector:
    """Loop detector class for detecting loop closures in image sequences"""
    
    def __init__(self, image_dir, 
                 output="loop_closures.txt",
                 config=None):
        """Initialize the loop detector
        
        Args:
            image_dir: Directory path containing images
            ckpt_path: Model checkpoint path
            image_size: Image resize dimensions [height width]
            batch_size: Batch size for processing
            similarity_threshold: Similarity threshold for loop closure
            top_k: Number of nearest neighbors to check for each image
            use_nms: Whether to use Non-Maximum Suppression (NMS) filtering
            nms_threshold: NMS threshold for minimum frame difference between loop pairs
            output: Output file path
        """
        self.config = config
        self.image_dir = image_dir
        self.ckpt_path = self.config['Weights']['SALAD']
        self.image_size = self.config['Loop']['SALAD']['image_size']
        self.batch_size = self.config['Loop']['SALAD']['batch_size']
        self.similarity_threshold = self.config['Loop']['SALAD']['similarity_threshold']
        self.top_k = self.config['Loop']['SALAD']['top_k']
        self.use_nms = self.config['Loop']['SALAD']['use_nms']
        self.nms_threshold = self.config['Loop']['SALAD']['nms_threshold']
        self.output = output
        
        self.model = None
        self.device = None
        self.image_paths = None
        self.descriptors = None
        self.loop_closures = None
    
    def _input_transform(self, image_size=None):
        """Create image transformation function"""
        MEAN=[0.485, 0.456, 0.406]; STD=[0.229, 0.224, 0.225]
        if image_size:
            return T.Compose([
                T.Resize(image_size, interpolation=T.InterpolationMode.BILINEAR),
                T.ToTensor(),
                T.Normalize(mean=MEAN, std=STD)
            ])
        else:
            return T.Compose([
                T.ToTensor(),
                T.Normalize(mean=MEAN, std=STD)
            ])
    
    def load_model(self):
        """Load model"""
        model = VPRModel(
            backbone_arch='dinov2_vitb14',
            backbone_config={
                'num_trainable_blocks': 4,
                'return_token': True,
                'norm_layer': True,
            },
            agg_arch='SALAD',
            agg_config={
                'num_channels': 768,
                'num_clusters': 64,
                'cluster_dim': 128,
                'token_dim': 256,
            },
            vggt_long_config=self.config
        )

        model.load_state_dict(torch.load(self.ckpt_path))
        model = model.eval()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        print(f"Model loaded: {self.ckpt_path}")
        
        self.model = model
        self.device = device
        return model, device
    
    def get_image_paths(self):
        """Get paths of all image files in directory"""
        image_extensions = [".jpg", ".jpeg", ".png"]
        image_paths = []
        
        for ext in image_extensions:
            image_paths.extend(list(Path(self.image_dir).glob(f"*{ext}")))
            image_paths.extend(list(Path(self.image_dir).glob(f"*{ext.upper()}")))
        
        image_paths = sorted(image_paths)
        self.image_paths = image_paths
        return image_paths
    
    def extract_descriptors(self):
        """Extract image feature descriptors"""
        if self.model is None or self.device is None:
            self.load_model()
            
        if self.image_paths is None:
            self.get_image_paths()
            
        transform = self._input_transform(self.image_size)
        descriptors = []
        
        for i in tqdm(range(0, len(self.image_paths), self.batch_size), desc="Extracting features"):
            batch_paths = self.image_paths[i:i+self.batch_size]
            batch_imgs = []
            
            for path in batch_paths:
                try:
                    img = Image.open(path).convert('RGB')
                    img = transform(img)
                    batch_imgs.append(img)
                except Exception as e:
                    print(f"Error processing image {path}: {e}")
                    img = torch.zeros(3, 224, 224) if self.image_size is None else torch.zeros(3, self.image_size[0], self.image_size[1])
                    batch_imgs.append(img)
            
            batch_tensor = torch.stack(batch_imgs).to(self.device)
            
            with torch.no_grad():
                with torch.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu', dtype=torch.float16):
                    batch_descriptors = self.model(batch_tensor).cpu()
            
            descriptors.append(batch_descriptors)
        
        self.descriptors = torch.cat(descriptors)
        return self.descriptors
    
    def _apply_nms_filter(self, loop_closures, nms_threshold):
        """Apply Non-Maximum Suppression (NMS) filtering to loop pairs"""
        if not loop_closures or nms_threshold <= 0:
            return loop_closures

        sorted_loops = sorted(loop_closures, key=lambda x: x[2], reverse=True)
        filtered_loops = []
        suppressed = set()
        
        max_frame = max(max(idx1, idx2) for idx1, idx2, _ in loop_closures)
        
        for idx1, idx2, sim in sorted_loops:
            if idx1 in suppressed or idx2 in suppressed:
                continue
            
            filtered_loops.append((idx1, idx2, sim))
            
            suppress_range = set()
            
            start1 = max(0, idx1 - nms_threshold)
            end1 = min(idx1 + nms_threshold + 1, idx2) 
            suppress_range.update(range(start1, end1))
            
            start2 = max(idx1 + 1, idx2 - nms_threshold)
            end2 = min(idx2 + nms_threshold + 1, max_frame + 1)
            suppress_range.update(range(start2, end2))
            
            suppressed.update(suppress_range)
        
        return filtered_loops
    
    def _ensure_decending_order(self, tuples_list):
        return [(max(a, b), min(a, b), score) for a, b, score in tuples_list]
    
    def find_loop_closures(self):
        """Find loop closures"""
        if self.descriptors is None:
            self.extract_descriptors()
            
        import faiss
        
        embed_size = self.descriptors.shape[1]
        faiss_index = faiss.IndexFlatIP(embed_size)
        
        normalized_descriptors = self.descriptors.numpy()
        faiss_index.add(normalized_descriptors)
        
        similarities, indices = faiss_index.search(normalized_descriptors, self.top_k + 1)  # +1 because self is most similar
        
        loop_closures = []
        for i in range(len(self.descriptors)):
            # Skip first result (self)
            for j in range(1, self.top_k + 1):
                neighbor_idx = indices[i, j]
                similarity = similarities[i, j]

                if similarity > self.similarity_threshold and abs(i - neighbor_idx) > 10:
                    if i < neighbor_idx:
                        loop_closures.append((i, neighbor_idx, similarity))
                    else:
                        loop_closures.append((neighbor_idx, i, similarity))

        loop_closures = list(set(loop_closures))
        loop_closures.sort(key=lambda x: x[2], reverse=True)
        
        if self.use_nms and self.nms_threshold > 0:
            loop_closures = self._apply_nms_filter(loop_closures, self.nms_threshold)
        
        self.loop_closures = self._ensure_decending_order(loop_closures)
        return self.loop_closures


    def save_results(self):
        """Save loop detection results to file"""
        if self.loop_closures is None:
            self.find_loop_closures()
            
        with open(self.output, 'w', encoding='utf-8') as f:
            f.write("# Loop Detection Results (index1, index2, similarity)\n")
            if self.use_nms:
                f.write(f"# NMS filtering applied, threshold: {self.nms_threshold}\n")
            f.write("\n# Loop pairs:\n")
            for i, j, sim in self.loop_closures:
                f.write(f"{i}, {j}, {sim:.4f}\n")
            f.write("\n# Image path list:\n")
            for i, path in enumerate(self.image_paths):
                f.write(f"# {i}: {path}\n")
        
        print(f"Found {len(self.loop_closures)} loop pairs, results saved to {self.output}")
        if self.use_nms:
            print(f"NMS filtering applied, threshold: {self.nms_threshold}")
        
        if self.loop_closures:
            print("\nTop 10 loop pairs:")
            for i, (idx1, idx2, sim) in enumerate(self.loop_closures[:10]):
                print(f"{idx1}, {idx2}, similarity: {sim:.4f}")
                if i >= 9:
                    break

    def get_loop_list(self):
        return [(idx1, idx2) for idx1, idx2, _ in self.loop_closures]

    def run(self):
        """Run complete loop detection pipeline"""
        print('Loading model...')
        self.load_model()
        
        self.get_image_paths()
        if not self.image_paths:
            print(f"No image files found in {self.image_dir}")
            return
        
        print(f"Found {len(self.image_paths)} image files")
        
        self.extract_descriptors()
        
        self.find_loop_closures()
        
        self.save_results()
        
        return self.loop_closures


def main():
    parser = argparse.ArgumentParser(description="Loop detection using SALAD model")
    parser.add_argument("--image_dir", type=str, default="/media/deng/Data/KITTIdataset/data_odometry_color/dataset/sequences/00/image_2", help="Directory path containing images")
    parser.add_argument("--ckpt_path", type=str, default="./weights/dino_salad.ckpt", help="Model checkpoint path")
    parser.add_argument("--image_size", nargs=2, type=int, default=[336, 336], help="Image resize dimensions [height width]")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for processing")
    parser.add_argument("--similarity_threshold", type=float, default=0.7, help="Similarity threshold for loop closure")
    parser.add_argument("--top_k", type=int, default=5, help="Number of nearest neighbors to check for each image")
    parser.add_argument("--output", type=str, default="loop_closures.txt", help="Output file path")
    parser.add_argument("--use_nms", action="store_true", default=True, help="Whether to use Non-Maximum Suppression (NMS) filtering")
    parser.add_argument("--nms_threshold", type=int, default=25, help="NMS threshold for minimum frame difference between loop pairs")
    
    args = parser.parse_args()
    
    detector = LoopDetector(
        image_dir=args.image_dir,
        ckpt_path=args.ckpt_path,
        image_size=args.image_size,
        batch_size=args.batch_size,
        similarity_threshold=args.similarity_threshold,
        top_k=args.top_k,
        use_nms=args.use_nms,
        nms_threshold=args.nms_threshold,
        output=args.output
    )
    
    detector.run()


if __name__ == "__main__":
    main()
    

