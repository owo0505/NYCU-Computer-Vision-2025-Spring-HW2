import os
import json
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torchvision
import torchvision.transforms as T
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator


class DigitTestDataset(torch.utils.data.Dataset):
    def __init__(self, images_dir, transforms=None):
        self.images_dir = images_dir
        self.transforms = transforms
        self.images = []
        for filename in sorted(os.listdir(images_dir)):
            if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
                self.images.append({
                    'id': int(os.path.splitext(filename)[0]),
                    'file_name': filename
                })

    def __getitem__(self, idx):
        img_info = self.images[idx]
        img_path = os.path.join(self.images_dir, img_info['file_name'])
        img = Image.open(img_path).convert("RGB")
        img_id = img_info['id']
        if self.transforms is not None:
            img = self.transforms(img)
        target = {'image_id': torch.tensor([img_id])}
        return img, target

    def __len__(self):
        return len(self.images)


def collate_fn(batch):
    return tuple(zip(*batch))


def get_transform():
    return T.Compose([T.ToTensor()])


def get_model(num_classes, model_path, device):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(
        pretrained=False
    )

    anchor_generator = AnchorGenerator(
        sizes=((8,), (16,), (32,), (64,), (96,)),
        aspect_ratios=((0.5, 1.0, 2.0),) * 5
    )
    model.rpn.anchor_generator = anchor_generator

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model


def soft_nms(boxes, scores, sigma=0.5, score_thresh=0.001):
    N = boxes.shape[0]
    for i in range(N):
        max_pos = i + np.argmax(scores[i:])
        boxes[[i, max_pos]] = boxes[[max_pos, i]]
        scores[[i, max_pos]] = scores[[max_pos, i]]

        box_i = boxes[i]
        area_i = (box_i[2] - box_i[0]) * (box_i[3] - box_i[1])
        for j in range(i+1, N):
            box_j = boxes[j]
            xx1 = max(box_i[0], box_j[0])
            yy1 = max(box_i[1], box_j[1])
            xx2 = min(box_i[2], box_j[2])
            yy2 = min(box_i[3], box_j[3])
            w = max(0.0, xx2 - xx1)
            h = max(0.0, yy2 - yy1)
            inter = w * h
            area_j = (box_j[2] - box_j[0]) * (box_j[3] - box_j[1])
            union = area_i + area_j - inter
            iou = inter / union if union > 0 else 0.0

            weight = np.exp(-(iou ** 2) / sigma)
            scores[j] = scores[j] * weight

    keep = np.where(scores > score_thresh)[0]
    return keep


def main():
    test_images_dir = "/path/to/test"
    num_classes = 11
    model_path = "/path/to/model.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_test = DigitTestDataset(test_images_dir,
                                    transforms=get_transform())
    data_loader_test = torch.utils.data.DataLoader(dataset_test,
                                                   batch_size=1,
                                                   shuffle=False,
                                                   num_workers=4,
                                                   collate_fn=collate_fn)

    model = get_model(num_classes, model_path, device)

    task1_results = []
    task2_results = []

    for images, targets in tqdm(data_loader_test,
                                desc="Predicting on test set"):
        image = images[0].to(device)
        image_id = int(targets[0]['image_id'].item())
        outputs = model([image])[0]

        boxes_all = outputs['boxes'].detach().cpu().numpy()
        scores_all = outputs['scores'].detach().cpu().numpy()
        labels_all = outputs['labels'].detach().cpu().numpy()

        softnms_keep = []
        unique_labels = np.unique(labels_all)
        for cl in unique_labels:
            inds = np.where(labels_all == cl)[0]
            if len(inds) == 0:
                continue
            boxes_cls = boxes_all[inds]
            scores_cls = scores_all[inds]
            keep_inds = soft_nms(boxes_cls, scores_cls,
                                 sigma=0.5, score_thresh=0.001)
            softnms_keep.extend(inds[keep_inds])
        softnms_keep = np.array(softnms_keep, dtype=np.int64)

        boxes = boxes_all[softnms_keep]
        scores = scores_all[softnms_keep]
        labels = labels_all[softnms_keep]

        keep = scores >= 0.3
        boxes = boxes[keep]
        scores = scores[keep]
        labels = labels[keep]

        for box, score, label in zip(boxes, scores, labels):
            x_min, y_min, x_max, y_max = box
            bbox = [float(x_min), float(y_min),
                    float(x_max - x_min), float(y_max - y_min)]
            task1_results.append({
                "image_id": image_id,
                "bbox": bbox,
                "score": float(score),
                "category_id": int(label)
            })

        keep = scores >= 0.7
        boxes = boxes[keep]
        scores = scores[keep]
        labels = labels[keep]
        if len(boxes) == 0:
            pred_digit = -1
        else:
            sorted_indices = boxes[:, 0].argsort()
            pred_digits = [str(int(labels[i]) - 1) for i in sorted_indices]
            pred_digit = int("".join(pred_digits))
        task2_results.append({
            "image_id": image_id,
            "pred_label": pred_digit
        })

    with open("pred.json", 'w') as f:
        json.dump(task1_results, f)
    print("Task 1 results saved to pred.json")

    df = pd.DataFrame(task2_results)
    df.to_csv("pred.csv", index=False)
    print("Task 2 results saved to pred.csv")


if __name__ == "__main__":
    main()
