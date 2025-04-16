import os
import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

import torch
import torch.utils.data
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator
import torchvision.transforms as T


class DigitCocoDataset(torch.utils.data.Dataset):
    def __init__(self, images_dir, target_json, transforms=None):
        self.images_dir = images_dir
        self.transforms = transforms
        with open(target_json, 'r') as f:
            self.coco = json.load(f)
        self.images = self.coco['images']
        self.annotations = self.coco['annotations']
        self.imgid_to_anns = {}
        for ann in self.annotations:
            self.imgid_to_anns.setdefault(ann['image_id'], []).append(ann)

    def __getitem__(self, idx):
        img_info = self.images[idx]
        img_path = os.path.join(self.images_dir, img_info['file_name'])
        img = Image.open(img_path).convert("RGB")
        img_id = img_info['id']

        anns = self.imgid_to_anns.get(img_id, [])
        boxes = []
        labels = []
        areas = []
        iscrowd = []
        for ann in anns:
            box = ann['bbox']
            x_min, y_min, w, h = box
            x_max = x_min + w
            y_max = y_min + h
            boxes.append([x_min, y_min, x_max, y_max])
            labels.append(ann['category_id'])
            areas.append(ann.get('area', w * h))
            iscrowd.append(ann.get('iscrowd', 0))
        target = {}
        target['boxes'] = torch.as_tensor(boxes, dtype=torch.float32)
        target['labels'] = torch.as_tensor(labels, dtype=torch.int64)
        target['image_id'] = torch.tensor([img_id])
        target['area'] = torch.as_tensor(areas, dtype=torch.float32)
        target['iscrowd'] = torch.as_tensor(iscrowd, dtype=torch.int64)

        if self.transforms is not None:
            img = self.transforms(img)
        return img, target

    def __len__(self):
        return len(self.images)


def collate_fn(batch):
    return tuple(zip(*batch))


def get_transform(train):
    transforms = [T.ToTensor()]
    if train:
        transforms.append(T.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.1
        ))
    return T.Compose(transforms)


def get_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(
        pretrained=True
    )

    anchor_generator = AnchorGenerator(
        sizes=((8,), (16,), (32,), (64,), (96,)),
        aspect_ratios=((0.5, 1.0, 2.0),) * 5
    )
    model.rpn.anchor_generator = anchor_generator

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    running_loss = 0
    for images, targets in tqdm(data_loader):
        images = [image.to(device) for image in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        running_loss += losses.item()

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
    avg_loss = running_loss / len(data_loader)
    print(f"Epoch {epoch} training loss: {avg_loss:.4f}")
    return avg_loss


@torch.no_grad()
def evaluate_loss(model, data_loader, device):
    model.train()
    total_loss = 0
    count = 0
    for images, targets in data_loader:
        images = [image.to(device) for image in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        total_loss += losses.item()
        count += 1
    avg_loss = total_loss / count if count > 0 else 0
    print(f"Validation loss: {avg_loss:.4f}")
    model.eval()
    return avg_loss


def soft_nms(boxes, scores, sigma=0.5, score_thresh=0.001):
    """
    boxes: (N, 4) 的 numpy 陣列，格式為 [x1, y1, x2, y2]
    scores: (N,) 的 numpy 陣列
    sigma: 控制衰減幅度的參數
    iou_thresh: IoU 閥值，當 IoU 超過此值時，對分數進行衰減
    score_thresh: 最終分數下限
    返回：保留的索引列表（根據 soft-NMS 後的結果）
    """
    N = boxes.shape[0]
    for i in range(N):
        max_pos = i + np.argmax(scores[i:])
        # 交換 i 與 max_pos 的 box、score
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


@torch.no_grad()
def evaluate_map(model, data_loader, device, score_threshold=0.5):
    model.eval()
    coco_predictions = []
    for images, targets in data_loader:
        images = [image.to(device) for image in images]
        outputs = model(images)
        for target, output in zip(targets, outputs):
            img_id = int(target['image_id'].item())
            boxes_all = output['boxes'].detach().cpu().numpy()
            scores_all = output['scores'].detach().cpu().numpy()
            labels_all = output['labels'].detach().cpu().numpy()

            softnms_keep_indices = []
            unique_labels = np.unique(labels_all)
            for cl in unique_labels:
                inds = np.where(labels_all == cl)[0]
                if len(inds) == 0:
                    continue
                boxes_cls = boxes_all[inds]
                scores_cls = scores_all[inds]
                keep_inds = soft_nms(boxes_cls, scores_cls, sigma=0.5,
                                     score_thresh=score_threshold)
                softnms_keep_indices.extend(inds[keep_inds])
            softnms_keep_indices = np.array(softnms_keep_indices,
                                            dtype=np.int64)

            boxes = boxes_all[softnms_keep_indices]
            scores = scores_all[softnms_keep_indices]
            labels = labels_all[softnms_keep_indices]

            for box, score, label in zip(boxes, scores, labels):
                x_min, y_min, x_max, y_max = box
                bbox = [float(x_min), float(y_min),
                        float(x_max - x_min), float(y_max - y_min)]
                coco_predictions.append({
                    "image_id": img_id,
                    "category_id": int(label),
                    "bbox": bbox,
                    "score": float(score)
                })

    gt_path = getattr(
        data_loader.dataset,
        "coco_path",
        "data/val/annotations.json",
    )
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    coco_gt = COCO(gt_path)
    coco_dt = coco_gt.loadRes(coco_predictions)

    coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    mAP = coco_eval.stats[0]
    print(f"mAP: {mAP:.4f}")
    return mAP


def main():
    train_images_dir = "/kaggle/input/dl-hw2/nycu-hw2-data/train"
    train_json = "/kaggle/input/dl-hw2/nycu-hw2-data/train.json"
    val_images_dir = "/kaggle/input/dl-hw2/nycu-hw2-data/valid"
    val_json = "/kaggle/input/dl-hw2/nycu-hw2-data/valid.json"
    num_classes = 11
    num_epochs = 6
    batch_size = 4
    lr = 0.005
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    score_threshold = 0.5

    dataset_train = DigitCocoDataset(train_images_dir, train_json,
                                     transforms=get_transform(train=True))
    dataset_val = DigitCocoDataset(val_images_dir, val_json,
                                   transforms=get_transform(train=False))
    dataset_val.coco_path = val_json

    data_loader_train = torch.utils.data.DataLoader(dataset_train,
                                                    batch_size=batch_size,
                                                    shuffle=True,
                                                    num_workers=4,
                                                    collate_fn=collate_fn)

    data_loader_val = torch.utils.data.DataLoader(dataset_val,
                                                  batch_size=1,
                                                  shuffle=False,
                                                  num_workers=4,
                                                  collate_fn=collate_fn)

    model = get_model(num_classes)
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9,
                                weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=4, gamma=0.1)

    train_losses = []
    val_losses = []
    map_scores = []

    for epoch in range(1, num_epochs + 1):
        print(f"\n--- Epoch {epoch} ---")
        train_loss = train_one_epoch(model, optimizer,
                                     data_loader_train, device, epoch)
        val_loss = evaluate_loss(model, data_loader_val, device)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        scheduler.step()

        mAP = evaluate_map(model, data_loader_val, device, score_threshold)
        map_scores.append(mAP)

        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict()
        }
        torch.save(checkpoint, f"epoch_{epoch}.pth")
        print(f"New model saved at epoch {epoch} with val loss {val_loss:.4f}")

    epochs = range(1, num_epochs + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label="Train Loss", marker="o")
    plt.plot(epochs, val_losses, label="Validation Loss", marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Learning Curve")
    plt.legend()
    plt.grid(True)
    plt.savefig("learning_curve.png")
    plt.show()
    print("Learning curve saved as learning_curve.png")

    mAP_epochs = [
        epoch
        for epoch, m in zip(epochs, map_scores)
        if m is not None
    ]
    mAP_values = [m for m in map_scores if m is not None]
    if mAP_epochs:
        plt.figure(figsize=(8, 6))
        plt.plot(mAP_epochs, mAP_values, label="mAP",
                 marker="o", color="green")
        plt.xlabel("Epoch")
        plt.ylabel("mAP")
        plt.title("mAP on Validation Set")
        plt.legend()
        plt.grid(True)
        plt.savefig("mAP_curve.png")
        plt.show()
        print("mAP curve saved as mAP_curve.png")


if __name__ == "__main__":
    main()
