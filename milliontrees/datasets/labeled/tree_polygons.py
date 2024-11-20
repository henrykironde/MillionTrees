from ..base.labeled import LabeledDataset
from PIL import Image, ImageDraw
import numpy as np
import torch
from shapely.wkt import loads as from_wkt
from torchvision.transforms import transforms
from torchvision.ops import masks_to_boxes
from ..utils import Mask, BoundingBoxes

class TreePolygonsDataset(LabeledDataset):
    """TreePolygons labeled dataset."""
    
    _dataset_name = "TreePolygons"
    _versions_dict = {
        "1.0": {
            "download_url": "",
            "compressed_size": 0,
        }
    }
    
    def _process_labels(self, df: pd.DataFrame) -> None:
        for i in range(len(df)):
            df.loc[i, 'polygon'] = from_wkt(df.loc[i, 'polygon'])
        self._y_array = list(df['polygon'].values)
        self._n_classes = 1
        self._y_size = 4
    def __getitem__(self, idx):
        # Any transformations are handled by the WILDSSubset
        # since different subsets (e.g., train vs test) might have different transforms
        x = self.get_input(idx)
        y_indices = self._input_lookup[self._input_array[idx]]
        y_polygons = [self._y_array[i] for i in y_indices]
        mask_imgs = [self.create_polygon_mask(x.shape[-2:], y_polygon) for y_polygon in y_polygons]
        masks = torch.concat([Mask(transforms.PILToTensor()(mask_img), dtype=torch.bool) for mask_img in mask_imgs])
        bboxes = BoundingBoxes(data=masks_to_boxes(masks), format='xyxy', canvas_size=x.size[::-1])

        metadata = self.metadata_array[idx]
        targets = {"y": masks, "bboxes": bboxes, "labels": np.zeros(len(masks), dtype=int)}

        return metadata, x, targets

    def create_polygon_mask(self, image_size, vertices):
        """
        Create a grayscale image with a white polygonal area on a black background.

        Parameters:
        - image_size (tuple): A tuple representing the dimensions (width, height) of the image.
        - vertices (list): A list of tuples, each containing the x, y coordinates of a vertex
                            of the polygon. Vertices should be in clockwise or counter-clockwise order.

        Returns:
        - PIL.Image.Image: A PIL Image object containing the polygonal mask.
        """
        # Create a new black image with the given dimensions
        mask_img = Image.new('L', image_size, 0)

        # Draw the polygon on the image. The area inside the polygon will be white (255).
        # Get the coordinates of the polygon vertices
        polygon_coords = [(int(vertex[0]), int(vertex[1]))
                          for vertex in vertices.exterior.coords._coords]

        # Draw the polygon on the image. The area inside the polygon will be white (255).
        ImageDraw.Draw(mask_img, 'L').polygon(polygon_coords, fill=(255))

        # Return the image with the drawn polygon as numpy array
        mask_img = np.array(mask_img)

        return mask_img

    def eval(self, y_pred, y_true, metadata, prediction_fn=None):
        """Computes all evaluation metrics.

        Args:
            - y_pred (Tensor): Predictions from a model. By default, they are predicted labels (LongTensor).
                               But they can also be other model outputs such that prediction_fn(y_pred)
                               are predicted labels.
            - y_true (LongTensor): Ground-truth labels
            - metadata (Tensor): Metadata
            - prediction_fn (function): A function that turns y_pred into predicted labels
        Output:
            - results (dictionary): Dictionary of evaluation metrics
            - results_str (str): String summarizing the evaluation metrics
        """
        metrics = [
            Accuracy(prediction_fn=prediction_fn),
            Recall(prediction_fn=prediction_fn, average='macro'),
            F1(prediction_fn=prediction_fn, average='macro'),
        ]

        results = {}

        for i in range(len(metrics)):
            results.update({
                **metrics[i].compute(y_pred, y_true),
            })

        results_str = (
            f"Average acc: {results[metrics[0].agg_metric_field]:.3f}\n"
            f"Recall macro: {results[metrics[1].agg_metric_field]:.3f}\n"
            f"F1 macro: {results[metrics[2].agg_metric_field]:.3f}\n")

        return results, results_str


    def get_input(self, idx):
        """
        Args:
            - idx (int): Index of a data point
        Output:
            - x (np.ndarray): Input features of the idx-th data point
        """
        # All images are in the images folder
        img_path = os.path.join(self._data_dir / 'images' / self._input_array[idx])
        img = Image.open(img_path)
        img = np.array(img.convert('RGB'))/255
        img = np.array(img, dtype=np.float32)

        return img
    
    def _transform_(self):
        self.transform = A.Compose([
            A.Resize(height=448, width=448, p=1.0),
            ToTensorV2()
            ])
        
        return self.transform