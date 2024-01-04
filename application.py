import argparse
import logging
import os
import pandas as pd
import torch
import mmcv

from annoy import AnnoyIndex
from flask import Flask, request, render_template
from pathlib import Path
from mmdet.apis import DetInferencer
from PIL import Image
from scipy.stats import percentileofscore

# get working directory
wdir = Path(os.getcwd())
logging.basicConfig(level=logging.INFO)


def init_inferencer(device):
    """Initialize the DetInferencer"""

    inferencer = DetInferencer(
        # Deformable-DETR (DETR is faster)
        model=str(
            wdir
            / "data/V3Det/checkpoints/configs/v3det/deformable-detr-refine-twostage_swin_16xb2_sample1e-3_v3det_50e.py"
        ),
        weights=str(wdir / "data/V3Det/checkpoints/Deformable_DETR_V3Det_SwinB.pth"),
        device=device,
    )

    return inferencer


def get_metrics_dist_v3d():
    """get the metrics distribution of V3D
    Return
        metrics_dist_v3d: a DataFrame with the following columns:
            - img_name (str): the image name
            - mni (float): the mni of the project
            - readability (float): the readability of the project
            - uniqueness (float): the uniqueness of the project
    """
    metrics_dist_v3d = pd.read_feather("data/metrics_dist_p50_allv3d.feather")
    metrics_dist_v3d["uniqueness"] = metrics_dist_v3d["uniqueness"] * 1000

    return metrics_dist_v3d


def get_obj_freq(context, prob_threshold, obj_df):
    """Get the object-level frequency
    Args:
        context: str, "general" or "user"

    Returns:
        a DataFrame that contains the following columns:
            - label (int): the label of the object
            - freq (float): the normalized frequency of the object
    """
    # for V3D context
    if context == "general":
        freq = pd.read_feather(
            # for v3d, we only use one threahold (0.5)
            "data/freq_v3d_p50.feather",
            columns=["label", "freq"],
        )

        return freq

    # for user context
    elif context == "user":
        freq = (
            obj_df.loc[obj_df.score >= prob_threshold]
            .groupby(["label"])
            .size()
            .reset_index(name="freq")
        )
        freq["freq"] = freq["freq"] / freq["freq"].max()

        return freq


def get_obj_mni(context, obj_df, prob_threshold, img_dir, inferencer, device, k=100):
    """get the object-level MNI
    Args:
        context: str, "general" or "user"
        k: int, the number of nearest neighbors to use for computing MNI

    Returns:
        obj_mni: a DataFrame that contains the mni of each object

    """
    # V3D context
    if context == "general":
        # load pre-computed object-level MNI on V3D
        return pd.read_feather("data/obj_mni_v3d_k100_p50.feather")

    # user context
    elif context == "user":
        img_feat_map = get_img_feat_map(img_dir, inferencer, device)

        # check if the number of images is large enough
        if len(img_feat_map) <= 50:
            logging.warning(
                f"For a meaning Concreteness (MNI) score, we recommend you provide at least 50 images. You only provided {len(img_feat_map)} images."
            )

        # get image names
        img_names = list(img_feat_map.keys())

        # initialize annoy tree "t"
        t = AnnoyIndex(1024, "angular")

        # a map from img_name to an int index
        img2id = {}

        for i, (img, vec) in enumerate(img_feat_map.items()):
            # create a map from img_name to an int index
            img2id[img] = i

            # add image embeddings to annoy tree
            t.add_item(i, vec)

        # build annoy tree

        # set seed
        t.set_seed(42)

        # n_trees is the number of hyperplans splits we want to build
        # 10 to 100 is a good starting point of n_trees, but since we only have 3k images
        # in kickstarter, we can build more trees (i.e., 1000)
        t.build(n_trees=100)

        # --- Compute object-level MNI  --- #

        # filter out objects with score < prob_threshold
        obj_df = obj_df[obj_df.score >= prob_threshold]

        # get a list of all unique objects, `objs`
        # each element in `objs` is an integer ID for an object category
        objs = obj_df.label.unique().tolist()
        logging.info(f"Number of unique objects: {len(objs)}")

        # get total number of images
        # the `V` in the formula, which is used to normalize MNI
        V = len(img_names)

        # initialize a dictionary to store final results
        mni_dict = {}

        # compute mni for each object category
        for i, obj in enumerate(objs):
            # `obj` is an integer ID for an object category

            # get a list of all image names that contain `obj`
            V_obj = obj_df[obj_df.label == obj].img_name.unique().tolist()

            # convert this list of img_names to a list of int indices
            # remember that `img2id` is a map from img_name to an int index
            V_obj = [img2id[img_name] for img_name in V_obj]
            V_obj = set(V_obj)

            # compute mni
            a = 0
            for v in V_obj:
                # `v` is an int index for an image

                # get a list of k nearest neighbors (named by its int indices)
                # for `v` (excluding `v` itself)
                NN_v = set(t.get_nns_by_item(v, k, search_k=-1)) - set([v])

                # get the number of images that contain `obj` and are also in `NN_v`
                a += len(V_obj.intersection(NN_v))

            # divide by the total number of images that contain `obj`
            mni_obj = a / len(V_obj)

            # normalize mni
            adj_mni = mni_obj / (len(V_obj) * k) * V

            mni_dict[obj] = adj_mni

        obj_mni_df = pd.DataFrame(mni_dict.items(), columns=["obj", "mni"])

        return obj_mni_df


def get_objects(img_dir, inferencer):
    """Identify the objects in the image
    Args:
        img_dir: str, the dir of the images

    Returns:
        a DataFrame with the following columns:
        - label (int): the label of the object
        - score: the confidence score of the object
        - size_ratio: the ratio of the size of the object to the size of the image
    """

    # get img_paths
    img_dir = Path(img_dir)
    img_paths = [
        file
        for file in img_dir.glob("**/*")
        if file.suffix.lower() in [".jpg", ".png", ".jpeg"]
    ]

    # Get the objects and feature map of the image
    obj_df = []
    for img_path in img_paths:
        # get objects
        objects = inferencer(str(img_path), show=False)["predictions"][0]

        # get size of each object
        obj_size = [w * h for x, y, w, h in objects["bboxes"]]

        # get the image size
        with Image.open(img_path) as img:
            w, h = img.size
            img_size = w * h

        # get the ratio of each object
        obj_size_ratio = [x / img_size for x in obj_size]

        # collect the results into a dataframe
        df = pd.DataFrame(
            {
                "img_name": img_path.stem,
                "label": objects["labels"],
                "score": objects["scores"],
                "size_ratio": obj_size_ratio,
            }
        )

        # append to obj_df
        obj_df.append(df)

    obj_df = pd.concat(obj_df, ignore_index=True)

    # the output label starts from 0, but V3D label starts from 1
    # so we add 1 to the label
    obj_df["label"] = obj_df.label.astype(int) + int(1)

    return obj_df


def get_img_feat_map(img_dir, inferencer, device):
    """Get the feature map of the image
    Args:
        img_dir: str, the dir of the images
        inferencer: the DetInferencer

    Returns:
        feat_map: dict of feature maps
    """
    # modify the model to remove the last laters (only keep the backbone)
    extractor = inferencer.model.backbone
    extractor.eval()

    # get img_paths
    img_dir = Path(img_dir)
    img_paths = [
        file
        for file in img_dir.glob("**/*")
        if file.suffix.lower() in [".jpg", ".png", ".jpeg"]
    ]

    # Get the feature map of the image
    img_feat = {}

    for img_path in img_paths:
        for out in inferencer.preprocess([mmcv.imread(img_path)]):
            # get the last feature map
            feature_map = extractor(
                out[1]["inputs"][0].float().unsqueeze(0).to(device)
            )[-1]

            # average pooling
            # unlike CNN, Transformer models don't pool. So we need to pool manually
            feature_map = torch.nn.AdaptiveAvgPool2d((1, 1))(feature_map)

            # flatten
            feature_map = feature_map.flatten().cpu().detach().numpy()

            # collect the results
            img_feat[img_path.stem] = feature_map

    return img_feat


def get_uniqueness(obj_df, freq, prob_threshold):
    """Calculate uniqueness of the image
    Args:
        obj_df (DataFrame): a DataFrame with the following columns:
            - label (int): the label of the object
            - score: the confidence score of the object
            - size_ratio: the ratio of the size of the object to the size of the image

        freq: a DataFrame that contains the frequency score of every object

    Returns:
        a DataFrame with the following columns:
            - img_name (str): name of the image
            - freq (float): frequency of the object in the general context
    """

    # we only keep objects with score >= prob_threshold
    obj_df = obj_df.loc[obj_df.score >= prob_threshold]

    # compute the freq of each project by aggregating the freq of each object in the project
    uniqueness = (
        obj_df.merge(freq, on="label", how="inner")
        .groupby(["img_name"])
        .agg({"freq": "mean"})
        # .rename(columns={"freq": "uniqueness"})
        # .reset_index()
    )

    return uniqueness


def get_readability(obj_df, prob_threshold):
    """Calculate gunning fog index of the image
    Args:
        obj_df (DataFrame): a DataFrame with the following columns:
            - label (int): the label of the object
            - score: the confidence score of the object
            - size_ratio: the ratio of the size of the object to the size of the image

    Returns:
        a DataFrame with the following columns:
            - name (str): name of the image
            - readability (float): gunning fog index of the image
    """

    # get object number and object size
    def _readability(group):
        # compute object number
        obj_num = len(group)

        # compute object size
        obj_size_lt_10 = sum(group.size_ratio <= 0.1)

        # get gunning fog index
        readability = 0.4 * (obj_num + 100 * obj_size_lt_10 / obj_num)

        # return
        return pd.Series(readability, index=["readability"])

    out = (
        obj_df.loc[obj_df.score >= prob_threshold]
        .groupby(["img_name"])
        .apply(_readability)
        .reset_index()
    )

    return out


def get_mni(obj_df, obj_mni, prob_threshold):
    """Calculate mni (concreteness) of the image
    Args:
        obj_df (DataFrame): a DataFrame with the following columns:
            - label (int): the label of the object
            - score: the confidence score of the object
            - size_ratio: the ratio of the size of the object to the size of the image

    Returns:
        a DataFrame with the following columns:
            - name (str): name of the image
            - mni (float): mni of the image
    """

    # we only keep objects with score >= prob_threshold
    obj_df = obj_df.loc[obj_df.score >= prob_threshold]

    # compute project-level MNI
    out = (
        obj_df.merge(obj_mni, left_on="label", right_on="obj", how="inner")
        .groupby(["img_name"])
        .agg({"mni": "mean"})
        .reset_index()
    )

    return out


def get_metrics(uniqueness, readability, mni, out_dir):
    """Combine all the metrics (uniqueness, gunning-fog, mni) into a dictionary
    Args:
        uniqueness: a DataFrame of uniqueness
        readability: a DataFrame of readability
        mni: a DataFrame of mni

    Returns:
        a DataFrame with all the metrics
    """

    # combine all the metrics
    metrics = (
        uniqueness.merge(readability, on="img_name", how="inner")
        .merge(mni, on="img_name", how="inner")
        .round(2)  # round to 2 digits
    )

    # save the metrics
    out_dir = Path(out_dir)
    if not out_dir.exists():
        out_dir.mkdir(parents=True)

    metrics.to_feather(f"{str(out_dir)}/image_metrics.feather")

    return metrics


def parse_args():
    # create parser
    parser = argparse.ArgumentParser()

    # parse arguments
    parser.add_argument(
        "--prob_threshold",
        type=float,
        default=0.1,
        help="Threshold for the object confidence in the user-uploaded image. Any objects with confidence below this threshold will be removed.",
    )
    parser.add_argument(
        "--context",
        type=str,
        default="user",
        help='"general" will use the V3D training data as the context; "user" will use the user-uploaded image as the context.',
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help='The device to use for inference. "cpu" for CPU and "cuda:0" for GPU.',
    )
    parser.add_argument(
        "--img_dir",
        type=str,
        help="The directory to save the uploaded images.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=".",
        help="The directory to save the outputs.",
    )

    # parse arguments and return
    return parser.parse_args()


def main():
    # parse arguments
    args = parse_args()

    # init inferencer
    inferencer = init_inferencer(args.device)

    # get v3d metrics distribution
    metrics_dist_v3d = get_metrics_dist_v3d()

    # detect objects
    obj_df = get_objects(args.img_dir, inferencer)

    # --- Uniqueness --- #

    # get object-level frequency
    obj_freq = get_obj_freq(args.context, args.prob_threshold, obj_df)

    # calculate uniqueness
    uniqueness = get_uniqueness(obj_df, obj_freq, args.prob_threshold)

    # --- Concreteness --- #

    # get object-level MNI
    obj_mni = get_obj_mni(
        args.context,
        obj_df,
        args.prob_threshold,
        args.img_dir,
        inferencer,
        args.device,
        k=100,
    )

    # calculate mni
    mni = get_mni(obj_df, obj_mni, args.prob_threshold)

    # --- Readability --- #
    readability = get_readability(obj_df, args.prob_threshold)

    # --- combine all metrics --- #
    img_metrics = get_metrics(
        uniqueness, readability, mni, args.out_dir
    )

    return img_metrics



if __name__ == "__main__":
    main()
