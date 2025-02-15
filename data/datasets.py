import os, torch, json

import pandas as pd
import numpy as np
import torch.utils.data as data
import pandas as pd
import numpy as np
import torch.utils.data as data
import sys

from typing import Callable, Dict, List, Tuple, Union
from pathlib import Path
from PIL import Image
from copy import deepcopy
from .constants import (
    DEFAULT_REFLACX_BOX_COORD_COLS,
    DEFAULT_REFLACX_BOX_FIX_COLS,
    DEFAULT_REFLACX_ALL_DISEASES,
    DEFAULT_REFLACX_LABEL_COLS,
    DEFAULT_REFLACX_PATH_COLS,
    DEFAULT_REFLACX_REPETITIVE_LABEL_MAP,
    SPREADSHEET_FOLDER,
)
from .helpers import map_target_to_device
from .fixation import get_fixations_dict_from_fixation_df, get_heatmap


def collate_fn(batch: Tuple) -> Tuple:
    return tuple(zip(*batch))


class ReflacxDataset(data.Dataset):
    """
    Class to load the preprocessed REFLACX master sheet. There `.csv` files are required to run this class.

    - `reflacx_for_eyetracking.csv'

    """

    def __init__(
        self,
        XAMI_MIMIC_PATH: str,
        with_fixations: bool = False,
        with_pupil: bool = False,
        with_1st_third_fixations: bool = False,
        with_2nd_third_fixations: bool = False,
        with_rad_silence: bool = False,
        with_rad_speaking: bool = False,
        bbox_to_mask: bool = False,
        split_str: str = None,
        transforms: Callable[[Image.Image, Dict], Tuple[torch.Tensor, Dict]] = None,
        dataset_mode: str = "normal",
        labels_cols: List[str] = DEFAULT_REFLACX_LABEL_COLS,
        all_disease_cols: List[str] = DEFAULT_REFLACX_ALL_DISEASES,
        repetitive_label_map: Dict[
            str, List[str]
        ] = DEFAULT_REFLACX_REPETITIVE_LABEL_MAP,
        box_fix_cols: List[str] = DEFAULT_REFLACX_BOX_FIX_COLS,
        box_coord_cols: List[str] = DEFAULT_REFLACX_BOX_COORD_COLS,
        path_cols: List[str] = DEFAULT_REFLACX_PATH_COLS,
        spreadsheets_folder=SPREADSHEET_FOLDER,
    ):
        # Data loading selections
        self.with_fixations = with_fixations
        self.with_pupil = with_pupil
        self.with_1st_third_fixations = with_1st_third_fixations
        self.with_2nd_third_fixations = with_2nd_third_fixations
        self.with_rad_silence = with_rad_silence
        self.with_rad_speaking = with_rad_speaking

        self.split_str: str = split_str

        # Image related
        self.transforms: Callable[
            [Image.Image, Dict], Tuple[torch.Tensor, Dict]
        ] = transforms
        self.path_cols: List[str] = path_cols

        # Labels
        self.labels_cols: List[str] = labels_cols
        self.all_disease_cols: List[str] = all_disease_cols
        self.repetitive_label_map: Dict[str, List[str]] = repetitive_label_map
        self.box_fix_cols: List[str] = box_fix_cols
        self.box_coord_cols: List[str] = box_coord_cols
        self.bbox_to_mask: bool = bbox_to_mask
        self.dataset_mode: str = dataset_mode

        # load dataframe
        self.df: pd.DataFrame = pd.read_csv(
            os.path.join(spreadsheets_folder, "reflacx_for_eyetracking.csv"), index_col=0
        )

        ## Split dataset.
        if not self.split_str is None:
            self.df: pd.DataFrame = self.df[self.df["split"] == self.split_str]

        ## replace the path with local mimic folder path.
        for p_col in path_cols:
            if p_col in self.df.columns:
                if p_col == "bbox_paths":

                    def apply_bbox_paths_transform(input_paths_str: str) -> List[str]:
                        input_paths_list: List[str] = json.loads(input_paths_str)
                        replaced_path_list: List[str] = [
                            p.replace("{XAMI_MIMIC_PATH}", XAMI_MIMIC_PATH)
                            for p in input_paths_list
                        ]
                        return replaced_path_list

                    apply_fn: Callable[
                        [str], List[str]
                    ] = lambda x: apply_bbox_paths_transform(x)

                else:
                    apply_fn: Callable[[str], str] = lambda x: str(
                        Path(x.replace("{XAMI_MIMIC_PATH}", XAMI_MIMIC_PATH))
                    )

                self.df[p_col] = self.df[p_col].apply(apply_fn)

        ## preprocessing data.
        self.preprocess_label()

        super(ReflacxDataset, self).__init__()

    def preprocess_label(self,):
        self.df[self.all_disease_cols] = self.df[self.all_disease_cols].gt(0)

    def load_image_array(self, image_path: str) -> np.ndarray:
        return np.asarray(Image.open(image_path))

    def plot_image_from_array(self, image_array: np.ndarray):
        im = Image.fromarray(image_array)
        im.show()

    def disease_to_idx(self, disease: str) -> int:
        if not disease in self.labels_cols:
            raise Exception("This disease is not the label.")

        if disease == "background":
            return 0

        return self.labels_cols.index(disease) + 1

    def label_idx_to_disease(self, idx: int) -> str:
        if idx == 0:
            return "background"

        if idx > len(self.labels_cols):
            return f"exceed label range :{idx}"

        return self.labels_cols[idx - 1]

    def __len__(self) -> int:
        return len(self.df)

    def generate_bboxes_df(self, ellipse_df: pd.DataFrame,) -> pd.DataFrame:
        boxes_df = ellipse_df[self.box_fix_cols]
        
        ## relabel repetitive columns.
        for k in self.repetitive_label_map.keys():
            boxes_df[k] = ellipse_df[
                [l for l in self.repetitive_label_map[k] if l in ellipse_df.columns]
            ].any(axis=1)
        
        ## filtering out the diseases not in the label_cols
        # boxes_df = boxes_df[boxes_df[self.labels_cols].any(axis=1)]

        # ## get labels
        # for index in boxes_df[self.labels_cols].index:  
        #     count_labels = 0
        #     for label in self.labels_cols:
        #         if boxes_df[self.labels_cols].at[index, label] and count_labels==0:
        #             count_labels +=1
        #             boxes_df.at[index, "label"] = label
        #         elif boxes_df[self.labels_cols].at[index, label] and count_labels>0:
        #             boxes_df = boxes_df.append({'xmin':boxes_df.at[index,'xmin'], 'ymin':boxes_df.at[index,'ymin'], 'xmax':boxes_df.at[index,'xmax'], 
        #                                         'ymax':boxes_df.at[index,'ymax'], 'xmax':boxes_df.at[index,'xmax'], 
        #                                         'certainty':boxes_df.at[index,'certainty'], 'label':label}, 
                                                # ignore_index=True)

        ## filtering out the diseases not in the label_cols
        boxes_df = boxes_df[boxes_df[self.labels_cols].any(axis=1)]
        ## get labels
        boxes_df["label"] = boxes_df[self.labels_cols].idxmax(axis=1)

        boxes_df = boxes_df[self.box_fix_cols + ["label"]]       

        return boxes_df

    def __getitem__(
        self, idx: int
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict],
        Tuple[torch.Tensor, Dict],
    ]:
        # find the df
        data: pd.Series = self.df.iloc[idx]

        # convert images to rgb
        img: Image = Image.open(data["image_path"]).convert("RGB")

        ## Get bounding boxes.
        bboxes_df = self.generate_bboxes_df(
            pd.read_csv(data["anomaly_location_ellipses_path"])
        )
        bboxes = torch.tensor(
            np.array(bboxes_df[self.box_coord_cols], dtype=float)
        )  # x1, y1, x2, y2

        ## Calculate area of boxes.
        area = (bboxes[:, 3] - bboxes[:, 1]) * (bboxes[:, 2] - bboxes[:, 0])

        labels = torch.tensor(
            np.array(bboxes_df["label"].apply(lambda l: self.disease_to_idx(l))),
            dtype=torch.int64,
        )

        image_id = torch.tensor([idx])
        num_objs = bboxes.shape[0]

        ## suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        # prepare all targets
        target = {}
        target["boxes"] = bboxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
        target["dicom_id"] = data["dicom_id"]
        target["image_path"] = data["image_path"]
        target["fixations_path"] = data["fixations_path"]

        if self.bbox_to_mask:
            # generate masks from bboxes
            masks = torch.zeros((num_objs, img.height, img.width), dtype=torch.uint8)
            for i, b in enumerate(bboxes):
                b = b.int()
                masks[i, b[1] : b[3], b[0] : b[2]] = 1
            target["masks"] = masks
            
        if self.with_fixations:
            # get fixations
            fix = get_heatmap(
                get_fixations_dict_from_fixation_df(pd.read_csv(data["fixations_path"])),
                (data["image_size_x"], data["image_size_y"]),
            ).astype(np.float32)

            img_t, target, fix_t = self.transforms(img, target, fix)

            return img_t, fix_t.repeat(3, 1, 1), target
        
        if self.with_pupil:
            # get pupil
            fix = get_heatmap(
                get_fixations_dict_from_fixation_df(pd.read_csv(data["fixations_path"])),
                (data["image_size_x"], data["image_size_y"]), pupil = True
            ).astype(np.float32)

            img_t, target, fix_t = self.transforms(img, target, fix)
            return img_t, fix_t.repeat(3, 1, 1), target

        if self.with_1st_third_fixations:
            # get first third fixations
            fix_dict = get_fixations_dict_from_fixation_df(pd.read_csv(data["fixations_path"]), first_third=True)

            fix = get_heatmap(fix_dict, (data["image_size_x"], data["image_size_y"])).astype(np.float32)

            img_t, target, fix_t = self.transforms(img, target, fix)
            return img_t, fix_t.repeat(3, 1, 1), target

        if self.with_2nd_third_fixations:
            # get second third fixations
            fix_dict = get_fixations_dict_from_fixation_df(pd.read_csv(data["fixations_path"]), second_third=True)

            fix = get_heatmap(fix_dict, (data["image_size_x"], data["image_size_y"])).astype(np.float32)

            img_t, target, fix_t = self.transforms(img, target, fix)
            return img_t, fix_t.repeat(3, 1, 1), target

        if self.with_rad_silence:
            # get fixations when radiologist is in silence

            #get timestamped transcriptions path
            trans_path = data["fixations_path"].replace('fixations', 'timestamps_transcription')

            fix_dict = get_fixations_dict_from_fixation_df(pd.read_csv(data["fixations_path"]), pd.read_csv(trans_path), rad_silence=True)

            fix = get_heatmap(fix_dict, (data["image_size_x"], data["image_size_y"])).astype(np.float32)

            img_t, target, fix_t = self.transforms(img, target, fix)
            return img_t, fix_t.repeat(3, 1, 1), target

        if self.with_rad_speaking:
            # get fixations when radiologist is speaking

            trans_path = data["fixations_path"].replace('fixations', 'timestamps_transcription')

            fix_dict = get_fixations_dict_from_fixation_df(pd.read_csv(trans_path), rad_speaking=True)

            fix = get_heatmap(fix_dict, (data["image_size_x"], data["image_size_y"])).astype(np.float32)

            img_t, target, fix_t = self.transforms(img, target, fix)
            return img_t, fix_t.repeat(3, 1, 1), target

        else:
            
            img_t, target = self.transforms(img, target)

            return img_t, target

    def prepare_input_from_data(
        self,
        data: Union[
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict],
            Tuple[torch.Tensor, Dict],
        ],
        device: str,
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict],
        Tuple[torch.Tensor, Dict],
    ]:
        if self.with_fixations or self.with_pupil or self.with_1st_third_fixations or self.with_rad_silence or self.with_rad_speaking or self.with_2nd_third_fixations:
            imgs, fixs, targets = data

            imgs = list(img.to(device) for img in imgs)
            fixs = list(fix.to(device) for fix in fixs)
            targets = [map_target_to_device(t, device) for t in targets]

            return (imgs, fixs, targets)

        imgs, targets = data

        imgs = list(img.to(device) for img in imgs)
        targets = [map_target_to_device(t, device) for t in targets]

        return (imgs, targets)

    def get_idxs_from_dicom_id(self, dicom_id: str) -> List[str]:
        return [
            self.df.index.get_loc(i)
            for i in self.df.index[self.df["dicom_id"].eq(dicom_id)]
        ]

    def get_image_path_from_dicom_id(self, dicom_id: str) -> List[str]:
        return self.df[self.df["dicom_id"] == dicom_id].iloc[0]["image_path"]

