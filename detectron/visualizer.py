import random
import cv2
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from dataset import get_logo_dicts


def visualize(cfg, amount=10):

        random.seed(42)
        predictor = DefaultPredictor(cfg)
        # predictor.model.load_state_dict(model.state_dict())
        dataset_dicts = get_logo_dicts(dirname="../data/datasets/openlogo", split="test", supervision="supervised_imageset")

        for data in random.sample(dataset_dicts, amount):
            img = cv2.imread(data["file_name"])
            outputs = predictor(img)
            vis = Visualizer(img[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TEST[0]), scale=1.2)
            vis = vis.draw_instance_predictions(outputs["instances"].to("cpu"))
            cv2.imshow("1", vis.get_image()[:, :, ::-1])
            cv2.waitKey(0)



