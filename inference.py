import os
import argparse
import sys
import pickle
import numpy as np
import cv2
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
import torch.nn.functional as F
from torch.utils.data import Dataset
import torch.distributed as dist
from torchvision import transforms
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
import segmentation_models_pytorch as smp
from transformers.models.maskformer.modeling_maskformer import dice_loss, sigmoid_focal_loss
from sam_of_theseus import build_sam_vit_h_c
from sam_of_theseus.replacement_scheduler import ConstantReplacementScheduler, LinearReplacementScheduler

# Add the SAM directory to the system path
sys.path.append("./segment-anything")
from segment_anything import sam_model_registry

# Import additional modules
from torch import nn
from copy import deepcopy
from PIL import Image
from lvis import LVIS


# Set the number of workers and GPUs
NUM_WORKERS = 0  # https://github.com/pytorch/pytorch/issues/42518
NUM_GPUS = torch.cuda.device_count()
DEVICE = 'cuda'

# Source: https://github.com/facebookresearch/detectron2/blob/main/detectron2/utils/comm.py
def get_world_size():
    """
    Get the world size in the distributed environment.
    Returns:
        int: The total number of ranks in the distributed environment.
    """
    # Return 1 if distributed training is not available
    if not dist.is_available():
        return 1
    # Return 1 if distributed training is not initialized
    if not dist.is_initialized():
        return 1
    # Return the total number of ranks in the current distributed environment
    return dist.get_world_size()

def all_gather(data):
    """
    Run all_gather on any serializable data (not necessarily tensors).
    Args:
        data: Any serializable object.
    Returns:
        list[data]: List of data collected from each rank.
    """
    # Get the total world size in the distributed environment
    world_size = get_world_size()
    if world_size == 1:
        return [data]

    # Serialize the data into a tensor
    buffer = pickle.dumps(data)
    # Create a byte storage object from the buffer and convert it to CUDA
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to("cuda")

    # Get the size of the tensor for each rank
    local_size = torch.LongTensor([tensor.numel()]).to("cuda")
    size_list = [torch.LongTensor([0]).to("cuda") for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    # Receive tensors from all ranks
    tensor_list = []
    for _ in size_list:
        tensor_list.append(torch.ByteTensor(size=(max_size,)).to("cuda"))
    if local_size != max_size:
        padding = torch.ByteTensor(size=(max_size - local_size,)).to("cuda")
        tensor = torch.cat((tensor, padding), dim=0)
    dist.all_gather(tensor_list, tensor)

    # Decode all received tensors into understandable Python objects and put them into a list
    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list



class LVISMaskDataset(Dataset):
    def __init__(self, data_root, split, image_size):
        self.data_root = data_root
        self.split = split
        self.image_size = image_size
        annotation = os.path.join(data_root, split, "_annotations.lvis.json")
        self.lvis = LVIS(annotation)
        # self.coco = Coco.from_coco_dict_or_path(annotation)

        # TODO: use ResizeLongestSide and pad to square
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.image_resize = transforms.Resize((image_size, image_size), interpolation=Image.BILINEAR)

    def __len__(self):
        return len(self.lvis.imgs)

    def __getitem__(self, index):

        lvis_images = self.lvis.imgs
        img_id = list(lvis_images.keys())[index]  # Get the ID of the corresponding image using the index
        # Initialize a variable to determine whether to skip the current image
        skip_current_image = False

        # Check if there are valid bboxes in the annotations of the current image; if not, skip it
        if all(annotation['bbox'] == [0, 0, 0, 0] for annotation in self.lvis.anns.values() if annotation['image_id'] == img_id):
            skip_current_image = True

        # If we need to skip the current image, recursively call __getitem__ to get the next image
        if skip_current_image:
            return self.__getitem__(index + 1)
        image_file_name = str(img_id).zfill(12) + ".jpg"
        image = Image.open(os.path.join(self.data_root, self.split, image_file_name)).convert("RGB")

        ratio_h = self.image_size / image.height
        ratio_w = self.image_size / image.width
        image = self.image_resize(image)
        image = self.to_tensor(image) 
        image = self.normalize(image)

        bboxes = []
        masks = []
        labels = []

        for annotation in self.lvis.anns.values():
            if annotation['image_id'] == img_id:  # Process annotations related to the current image only
                x, y, w, h = annotation['bbox']
                # Get scaled bbox in xyxy format
                bbox = [x * ratio_w, y * ratio_h, (x + w) * ratio_w, (y + h) * ratio_h]
                segmentation = annotation['segmentation']

                points = [np.array(point).reshape(-1, 2).round().astype(int) for point in segmentation]

                # Create a NumPy array of zeros with a size of [self.image_size, self.image_size] to store the boolean mask
                mask = np.zeros((self.image_size, self.image_size))

                # Use the cv2.fillPoly function to mark the region enclosed by the coordinate points as 1 (boolean True)
                mask = cv2.fillPoly(mask, points, 1)

                mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
                mask = (mask > 0.5).astype(np.uint8)
                label = annotation['category_id']
                bboxes.append(bbox)
                masks.append(mask)
                labels.append(label)

        bboxes = np.stack(bboxes, axis=0)
        masks = np.stack(masks, axis=0)
        labels = np.stack(labels, axis=0)

        return image, torch.tensor(bboxes), torch.tensor(masks).long()

    @classmethod
    
    def collate_fn(cls, batch):
        images, bboxes, masks = zip(*batch)
        images = torch.stack(images, dim=0)
        return images, bboxes, masks
    
class SAMFinetuner(pl.LightningModule):
    def __init__(
        self,
        model_type,
        checkpoint_path,
        batch_size=1,
        learning_rate=1e-4,
        weight_decay=1e-4,
        train_dataset=None,
        val_dataset=None,
        metrics_interval=10,
        pre_n_layers=32,
        scc_n_layers=16,
    ):
        super(SAMFinetuner, self).__init__()
        self.pre_n_layers = pre_n_layers
        self.scc_n_layers = scc_n_layers
        self.model_type = model_type
        self.orimodel = sam_model_registry[self.model_type](checkpoint=checkpoint_path)
        self.model = build_sam_vit_h_c(None)
        orimodel_params = self.orimodel.state_dict()
        model_params = self.model.state_dict()
        for name, param in orimodel_params.items():
            if name in model_params:
                model_params[name].copy_(param)
        self.model.load_state_dict(model_params)
        self.model.image_encoder.scc_blocks = nn.ModuleList(
            [deepcopy(self.orimodel.image_encoder.blocks[ix]) for ix in range(scc_n_layers)]
        )
        self.model.to(device=self.device)
        for param in self.model.image_encoder.parameters():
            param.requires_grad = False
        for param in self.model.mask_decoder.parameters():
            param.requires_grad = False
        for param in self.model.prompt_encoder.parameters():
            param.requires_grad = False
        for param in self.model.image_encoder.scc_blocks.parameters():
            param.requires_grad = True
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.train_metric = defaultdict(lambda: deque(maxlen=metrics_interval))
        self.metrics_interval = metrics_interval
        self.replacing_rate = 0.5
        self.steps_for_replacing = 0

    def forward(self, imgs, bboxes, labels):
        _, _, H, W = imgs.shape
        features = self.model.image_encoder(imgs)
        num_masks = sum([len(b) for b in bboxes])
        loss_focal = loss_dice = loss_iou = 0.
        predictions = []
        tp, fp, fn, tn = [], [], [], []
        for feature, bbox, label in zip(features, bboxes, labels):
            sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
                points=None,
                boxes=bbox,
                masks=None,
            )
            low_res_masks, iou_predictions = self.model.mask_decoder(
                image_embeddings=feature.unsqueeze(0),
                image_pe=self.model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )
            masks = F.interpolate(
                low_res_masks,
                (H, W),
                mode="bilinear",
                align_corners=False,
            )
            predictions.append(masks)
            batch_tp, batch_fp, batch_fn, batch_tn = smp.metrics.get_stats(
                masks,
                label.unsqueeze(1),
                mode='binary',
                threshold=0.5,
            )
            batch_iou = smp.metrics.iou_score(batch_tp, batch_fp, batch_fn, batch_tn)
            masks = masks.squeeze(1).flatten(1)
            label = label.flatten(1)
            loss_focal += sigmoid_focal_loss(masks, label.float(), num_masks)
            loss_dice += dice_loss(masks, label.float(), num_masks)
            loss_iou += F.mse_loss(iou_predictions, batch_iou, reduction='sum') / num_masks
            tp.append(batch_tp)
            fp.append(batch_fp)
            fn.append(batch_fn)
            tn.append(batch_tn)
        return {
            'loss': 20. * loss_focal + loss_dice + loss_iou,
            'loss_focal': loss_focal,
            'loss_dice': loss_dice,
            'loss_iou': loss_iou,
            'predictions': predictions,
            'tp': torch.cat(tp),
            'fp': torch.cat(fp),
            'fn': torch.cat(fn),
            'tn': torch.cat(tn),
        }

    def training_step(self, batch, batch_nb):
        imgs, bboxes, labels = batch
        outputs = self(imgs, bboxes, labels)
        for metric in ['tp', 'fp', 'fn', 'tn']:
            self.train_metric[metric].append(outputs[metric])
        step_metrics = [torch.cat(list(self.train_metric[metric])) for metric in ['tp', 'fp', 'fn', 'tn']]
        per_mask_iou = smp.metrics.iou_score(*step_metrics, reduction="micro-imagewise")
        metrics = {
            "loss": outputs["loss"],
            "loss_focal": outputs["loss_focal"],
            "loss_dice": outputs["loss_dice"],
            "loss_iou": outputs["loss_iou"],
            "train_per_mask_iou": per_mask_iou,
        }
        self.log_dict(metrics, prog_bar=True, rank_zero_only=True)
        return metrics

    def validation_step(self, batch, batch_nb):
        imgs, bboxes, labels = batch
        outputs = self(imgs, bboxes, labels)
        outputs.pop("predictions")
        return outputs

    def validation_epoch_end(self, outputs):
        if NUM_GPUS > 1:
            outputs = all_gather(outputs)
            outputs = [item for sublist in outputs for item in sublist]
        step_metrics = [
            torch.cat(list([x[metric].to(self.device) for x in outputs]))
            for metric in ['tp', 'fp', 'fn', 'tn']]
        per_mask_iou = smp.metrics.iou_score(*step_metrics, reduction="micro-imagewise")
        metrics = {"val_per_mask_iou": per_mask_iou}
        self.log_dict(metrics)
        return metrics

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        def warmup_step_lr_builder(warmup_steps, milestones, gamma):
            def warmup_step_lr(steps):
                if steps < warmup_steps:
                    lr_scale = (steps + 1.) / float(warmup_steps)
                else:
                    lr_scale = 1.
                    for milestone in sorted(milestones):
                        if steps >= milestone * self.trainer.estimated_stepping_batches:
                            lr_scale *= gamma
                return lr_scale
            return warmup_step_lr
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            opt,
            warmup_step_lr_builder(250, [0.66667, 0.86666], 0.1)
        )
        return {
            'optimizer': opt,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': "step",
                'frequency': 1,
            }
        }

    def train_dataloader(self):
        train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            collate_fn=self.train_dataset.collate_fn,
            batch_size=self.batch_size,
            num_workers=NUM_WORKERS,
            shuffle=True)
        return train_loader

    def val_dataloader(self):
        val_loader = torch.utils.data.DataLoader(
            self.val_dataset,
            collate_fn=self.val_dataset.collate_fn,
            batch_size=self.batch_size,
            num_workers=NUM_WORKERS,
            shuffle=False)
        return val_loader


def main():
    # Create an argument parser
    parser = argparse.ArgumentParser()

    # Add optional arguments
    parser.add_argument("--data_root", type=str, default='D:\DATASET\LVISv1', help="Dataset path")
    parser.add_argument("--model_type", type=str, default='vit_h', help="Model type", choices=['vit_h', 'vit_l', 'vit_b'])
    parser.add_argument("--checkpoint_path", default='D:\MODEL\sam_vit_h_4b8939.pth', type=str, help="SAM Checkpoint path")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--image_size", type=int, default=1024, help="Image size")
    parser.add_argument("--steps", type=int, default=5, help="Number of steps")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--metrics_interval", type=int, default=1, help="Log metrics interval")
    parser.add_argument("--output_dir", type=str, default="D:\MODEL\TRAIN\lvis", help="Model save path")
    parser.add_argument("--device", type=str, default="1", help="Device")

    # Parse command-line arguments
    args = parser.parse_args()

    train_dataset = LVISMaskDataset(data_root=args.data_root, split="val", image_size=args.image_size)
    val_dataset = LVISMaskDataset(data_root=args.data_root, split="train", image_size=args.image_size)

    # Create the model
    model = SAMFinetuner(
        args.model_type,
        args.checkpoint_path,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        metrics_interval=args.metrics_interval,
    )

    os.makedirs(args.output_dir + '/' + str(args.steps) + '/', exist_ok=True)

    # Create a list of callbacks
    callbacks = [
        LearningRateMonitor(logging_interval='step'),
        ModelCheckpoint(
            dirpath= args.output_dir + '/' + str(args.steps) + '/',
            filename='{step}-{val_per_mask_iou:.2f}',
            save_last=True,
            save_top_k=1,
            monitor="val_per_mask_iou",
            mode="max",
            save_weights_only=True,
            every_n_train_steps=args.metrics_interval,
        ),
    ]

    # Set Trainer parameters
    trainer = pl.Trainer(
        gpus=[int(args.device)],
        accelerator=DEVICE,
        precision=32,
        callbacks=callbacks,
        max_epochs=-1, 
        max_steps=args.steps,
        val_check_interval=args.metrics_interval,
        check_val_every_n_epoch=None,
        num_sanity_val_steps=0,
    )

    # Train the model
    trainer.fit(model)

if __name__ == "__main__":
    main()
