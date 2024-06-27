from vit_prisma.visualization.patch_level_logit_lens import (
    display_grid_on_image_with_heatmap,
)
from vit_prisma.utils.data_utils.imagenet_emoji import IMAGENET_EMOJI
from vit_prisma.utils.data_utils.imagenet_dict import IMAGENET_DICT
from collections import defaultdict
import torch
from vit_prisma.models.base_vit import HookedViT
from vit_prisma.prisma_tools.activation_cache import ActivationCache
from transformers import CLIPProcessor, CLIPModel, AutoTokenizer
from PIL import Image
from torchvision.transforms import Resize, Compose, ToTensor, Normalize
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from vit_prisma.prisma_tools.logit_lens import get_patch_logit_directions
from typing import Optional
import pandas as pd


def patch_forward(patch):
    patch = torch.sigmoid(patch)
    return patch


# Apply the adversarial_patch to the image
def apply_patch(image: torch.Tensor, patch: torch.Tensor):
    """
    :param image: torch.Tensor: Image tensor
    :param patch: torch.Tensor: Patch tensor (1, 3, 32, 32)
    """
    patch_size = patch.size(-1)
    image = image.clone()
    image[:, :, :patch_size, :patch_size] = patch_forward(patch)
    return image


class CLIPLens(object):
    MODEL_NAME = "wkcn/TinyCLIP-ViT-40M-32-Text-19M-LAION400M"

    def __init__(self):
        self.model = HookedViT.from_pretrained(
            self.MODEL_NAME, is_timm=False, is_clip=True
        )
        self.clip_model = CLIPModel.from_pretrained(self.MODEL_NAME)
        self.logit_scale = self.clip_model.logit_scale
        self.clip_processor = CLIPProcessor.from_pretrained(self.MODEL_NAME)
        # clip_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

        self.base_transform = Compose(
            [
                Resize((224, 224)),
                ToTensor(),
            ]
        )
        self.transform = Compose(
            [
                self.base_transform,
                Normalize(
                    mean=self.clip_processor.image_processor.image_mean,
                    std=self.clip_processor.image_processor.image_std,
                ),
            ]
        )
        self.transform = self.base_transform

    def compute_text_cache(
        self,
        entities: list[str],
        prompt_template: str = "a photo of a {}",
        emoji_mapping: dict[int, str] = {},
    ):
        prompts = [prompt_template.format(entity) for entity in entities]
        text_tokens = self.clip_processor(
            text=prompts, return_tensors="pt", padding=True
        )
        self.text_features = self.clip_model.get_text_features(**text_tokens)
        self.text_features /= self.text_features.norm(dim=-1, keepdim=True)
        self.entities = entities
        self.emoji_mapping = emoji_mapping

    def get_entity_idx(self, entity: str):
        return self.entities.index(entity)

    def get_entity(self, idx: int):
        return self.entities[idx]

    def get_logits_from_pre_proj_cls(self, pre_proj_cls_token: torch.Tensor):
        post_proj_cls_token = (
            pre_proj_cls_token @ self.model.head.W_H + self.model.head.b_H
        )
        post_proj_cls_token /= post_proj_cls_token.norm(dim=-1, keepdim=True)
        logits = self.logit_scale * post_proj_cls_token @ self.text_features.T
        return logits

    def compute_image_cache(
        self, image: Image.Image, patch: Optional[torch.Tensor] = None
    ):
        assert self.text_features is not None, "Text cache not computed."
        self.image = image.convert("RGB")
        self.image_tensor = self.transform(self.image).unsqueeze(0)

        # Show image patched with a random patch:
        if patch is not None:
            self.image_tensor = apply_patch(self.image_tensor, patch)

        self.image_tensor_unnormalized = self.base_transform(image)
        self.post_proj_cls_token, self.cache = self.model.run_with_cache(
            self.image_tensor
        )
        self.image_features = self.post_proj_cls_token
        self.logits: torch.Tensor = (
            self.logit_scale * self.image_features @ self.text_features.T
        )
        self.probs = self.logits.softmax(dim=-1).cpu().detach()

    def print_preds(self, print_idx: list[int] = [], top_k=10, layer_idx: int = -1):
        """
        Print top k predictions and optionally print the index of the predictions in the list

        :param image_tensor: torch.Tensor: Image tensor
        :param print_idx: list[int]: List of indices to print
        :param top_k: int: Number of top tokens to print
        """
        top_k = min(top_k, len(self.entities))
        cls_token_residual_stream = self.cache["resid_post", layer_idx][:, 0, :]
        pre_proj_cls_token = self.cache.apply_ln_to_stack(
            cls_token_residual_stream, layer=layer_idx, pos_slice=0
        )
        logits = self.get_logits_from_pre_proj_cls(pre_proj_cls_token)
        probs = logits.softmax(dim=-1).squeeze(0).cpu().detach().numpy()
        sorted_probs = np.sort(probs)[::-1]
        sorted_probs_args = np.argsort(probs)[::-1]
        output_str = ""
        data = []

        for i in range(top_k):
            index = sorted_probs_args[i]
            prob = sorted_probs[i]
            logit = self.logits[
                0, index
            ].item()  # Assuming you want to show the original logit value
            label = self.get_entity(index)  # Adjust based on your mapping

            rank_str = f"Top {i}th token."
            logit_str = f"Logit: {logit:.2f}"
            prob_str = f"Prob: {prob * 100:.2f}%"
            token_str = f"Label: |{label}|"
            output_str += f"{rank_str} {logit_str} {prob_str} {token_str}\n"
            data.append(
                {
                    "Rank": i,
                    "Logit": logit,
                    "Prob": prob,
                    "Label": label,
                }
            )

        if print_idx:
            for idx in print_idx:
                entity = self.get_entity(idx)
                rank = np.where(sorted_probs_args == idx)[0][0]
                output_str += f"Class Name: {entity} | Rank: {rank} | Index: {idx}\n"
                data.append(
                    {
                        "Rank": rank,
                        "Logit": self.logits[0, idx].item(),
                        "Prob": probs[idx],
                        "Label": entity,
                    }
                )
        print(output_str)
        return pd.DataFrame(data)

    def get_W_H(self, pre_proj_token: torch.Tensor) -> torch.Tensor:
        n = (pre_proj_token @ self.model.head.W_H + self.model.head.b_H).norm(
            dim=-1, keepdim=True
        )
        scaled_text_features = self.logit_scale * self.text_features.T
        return self.model.head.W_H / n @ scaled_text_features

    def get_b_H(self, pre_proj_token: torch.Tensor) -> torch.Tensor:
        n = (pre_proj_token @ self.model.head.W_H + self.model.head.b_H).norm(
            dim=-1, keepdim=True
        )
        scaled_text_features = self.logit_scale * self.text_features.T
        return self.model.head.b_H / n @ scaled_text_features

    def tokens_to_residual_directions(
        self, indices: np.ndarray, pre_proj_token: torch.Tensor
    ):
        """
        :param indices: np.ndarray: Indices of the tokens to get the residual directions for
        :param pre_proj_token: torch.Tensor: Pre projection token (after layer norm, before unembed)
        """
        new_W_H = self.get_W_H(pre_proj_token)
        return new_W_H.T[indices]

    def residual_stack_to_logit_v2(
        self, residual_stack: torch.Tensor, use_last_unembed: bool = True
    ):
        scaled_residual_stack = self.cache.apply_ln_to_stack(
            residual_stack, layer=-1, pos_slice=0
        )
        logit_predictions = []
        for layer_res in scaled_residual_stack:
            res = self.get_logits_from_pre_proj_cls(layer_res)
            logit_predictions.append(res)
        logit_predictions = torch.stack(logit_predictions, dim=-1).squeeze(0)
        return logit_predictions

    def get_patch_logit_directions_v2(
        self, indices, incl_mid=False, return_labels=True
    ):
        # check indices
        accumulated_residual, labels = self.cache.accumulated_resid(
            layer=-1, incl_mid=incl_mid, return_labels=True
        )
        result = self.residual_stack_to_logit_v2(accumulated_residual).unsqueeze(0)
        result = result.permute(0, 1, 3, 2)
        return result, labels

    def get_patch_logit_dictionary(
        self,
        patch_logit_directions: tuple[torch.Tensor, any],
        batch_idx=0,
        rank_label=None,
    ):
        patch_dictionary = defaultdict(list)
        # if tuple, get first entry
        if isinstance(patch_logit_directions, tuple):
            patch_logit_directions = patch_logit_directions[0]
        # Go through laeyrs of one batch
        for patch_idx, patches in enumerate(patch_logit_directions[batch_idx]):
            # Go through every patch and get max prediction
            for logits in patches:
                probs = torch.softmax(logits, dim=-1)
                # Get index of max prediction
                predicted_idx = int(torch.argmax(probs))
                logit = logits[predicted_idx].item()
                predicted_class_name = self.get_entity(predicted_idx)
                if rank_label:
                    # Where is the rank_label in the sorted list?
                    rank_index = self.get_entity_idx(rank_label)
                    sorted_list = torch.argsort(probs, descending=True)
                    rank = np.where(sorted_list == rank_index)[0][0]
                    patch_dictionary[patch_idx].append(
                        (logit, predicted_class_name, predicted_idx, rank)
                    )
                else:
                    patch_dictionary[patch_idx].append(
                        (logit, predicted_class_name, predicted_idx)
                    )
        return patch_dictionary

    def average_logit_value_across_all_classes(
        self,
        residual_stack: torch.Tensor,
        mean=True,
    ):
        scaled_residual_stack = self.cache.apply_ln_to_stack(
            residual_stack, layer=-1, pos_slice=0
        )
        logit_predictions = []
        for layer_res in scaled_residual_stack:
            res = self.get_logits_from_pre_proj_cls(layer_res)
            logit_predictions.append(res)
        logit_predictions = torch.stack(logit_predictions, dim=-1).squeeze(0)

        if mean:
            logit_predictions = logit_predictions.mean(axis=0)
        return logit_predictions

    def display_patch_logit_lens(
        self, no_bias: bool = True, layer_idx: int = -1, token_idx: int = 0
    ):
        n = len(self.entities)
        if no_bias:
            cls_token_residual_stream = self.cache["resid_post", layer_idx][
                :, token_idx, :
            ]
            pre_proj_cls_token = self.cache.apply_ln_to_stack(
                cls_token_residual_stream, layer=layer_idx, pos_slice=token_idx
            )
            all_answers = self.tokens_to_residual_directions(
                np.arange(n), pre_proj_token=pre_proj_cls_token
            )
            patch_logit_directions = get_patch_logit_directions(
                self.cache, all_answers, incl_mid=False
            )
        else:
            patch_logit_directions = self.get_patch_logit_directions_v2(
                np.arange(n), incl_mid=False
            )
        patch_dictionary = self.get_patch_logit_dictionary(
            patch_logit_directions, batch_idx=0
        )
        fig1 = display_patch_logit_lens(
            patch_dictionary,
            labels=patch_logit_directions[1],
            entity_to_emoji=self.emoji_mapping,
            width=1300,
            height=1000,
            emoji_size=22,
        )
        fig1.show()

    def display_logit_lens(
        self, layer_idx: int = -1, proj_layer_idx: int = -1, no_bias: bool = True
    ):
        n = len(self.entities)
        if no_bias:
            cls_token_residual_stream = self.cache["resid_post", proj_layer_idx][
                :, 0, :
            ]
            pre_proj_cls_token = self.cache.apply_ln_to_stack(
                cls_token_residual_stream, layer=proj_layer_idx, pos_slice=0
            )
            # pre_proj_cls_token = self.cache["ln_final.hook_normalized"][:, 0, :]
            all_answers = self.tokens_to_residual_directions(
                np.arange(n), pre_proj_cls_token
            )
            patch_logit_directions = get_patch_logit_directions(
                self.cache, all_answers, incl_mid=False
            )
        else:
            patch_logit_directions = self.get_patch_logit_directions_v2(
                np.arange(n), incl_mid=False
            )
        patch_dictionary_v2 = self.get_patch_logit_dictionary(
            patch_logit_directions, batch_idx=0
        )
        fig = display_grid_on_image_with_heatmap(
            self.image_tensor_unnormalized,
            patch_dictionary_v2,
            alpha_color=0.4,
            layer_idx=layer_idx,
            imagenet_class_to_emoji=self.emoji_mapping,
            emoji_font_size=30,
            return_graph=True,
        )
        fig.show()


def display_patch_logit_lens(
    patch_dictionary,
    entity_to_emoji: dict[int, str],
    width: int = 1000,
    height: int = 1200,
    emoji_size: int = 26,
    show_colorbar=True,
    labels=None,
):
    num_patches = len(patch_dictionary)

    # Assuming data_array_formatted is correctly shaped according to your data structure
    data_array_formatted = np.array(
        [
            [item[0] for item in list(patch_dictionary.values())[i]]
            for i in range(num_patches)
        ]
    )

    # Modify hover text generation based on whether labels are provided
    if labels:
        hover_text = [
            [
                f"{labels[j]}: {item[1]}"
                for j, item in enumerate(list(patch_dictionary.values())[i])
            ]
            for i in range(num_patches)
        ]
    else:
        hover_text = [
            [str(item[1]) for item in list(patch_dictionary.values())[i]]
            for i in range(num_patches)
        ]

    # Creating the interactive heatmap
    fig = go.Figure(
        data=go.Heatmap(
            z=data_array_formatted,
            x=list(patch_dictionary.keys())[:num_patches],
            y=[f"{i}" for i in range(data_array_formatted.shape[0])],  # Patch Number
            hoverongaps=False,
            colorbar=dict(title="Logit Value") if show_colorbar else None,
            text=hover_text,
            hoverinfo="text",
        )
    )

    # Initialize a list to hold annotations for emojis
    annotations = []

    # Calculate half the distance between cells in both x and y directions for annotation placement
    x_half_dist = 0.5
    y_half_dist = 0.2

    for i, patch in enumerate(patch_dictionary.values()):
        for j, items in enumerate(
            patch
        ):  # Extract class index directly from the patch_dictionary
            class_index = items[2]
            emoji = entity_to_emoji.get(
                class_index, ""
            )  # Use class index for emoji lookup, default to empty if not found
            if emoji:  # Add annotation if emoji is found
                annotations.append(
                    go.layout.Annotation(
                        x=j + x_half_dist,
                        y=i + y_half_dist,
                        text=emoji,
                        showarrow=False,
                        font=dict(color="white", size=emoji_size),
                    )
                )

    # Add annotations to the figure
    fig.update_layout(annotations=annotations)

    # Configure the layout of the figure
    fig.update_layout(
        title="Per-Patch Logit Lens",
        xaxis=dict(title="Layer Number"),
        yaxis=dict(title="Patch Number"),
        autosize=False,
        width=width,
        height=height,
    )
    fig.show()
    return fig


if __name__ == "__main__":
    clip_lens = CLIPLens()
    entities = list(IMAGENET_DICT.values())
    emoji_mapping = IMAGENET_EMOJI
    entities.append("")
    emoji_mapping[len(entities) - 1] = "📷"
    entities.append("cat and dog")
    emoji_mapping[len(entities) - 1] = "🐱🐶"
    # entities.append("cat and dog and mouse")
    # emoji_mapping[len(entities) - 1] = "🐱🐶🐭"
    # entities.append("cat and dog and mouse and serghei")
    # emoji_mapping[len(entities) - 1] = "🐱🐶🐭🧑‍🦱"
    entities.append("everything")
    emoji_mapping[len(entities) - 1] = "🌍"
    entities.append("nothing")
    emoji_mapping[len(entities) - 1] = "🚫"

    # entities = [" of a cat"," of a dog", " of a bird", " of a fish", " of a great white shark", " of a pig", " of a house", ""]
    # emoji_mapping = {
    #     0: "🐱",
    #     1: "🐶",
    #     2: "🐦",
    #     3: "🐟",
    #     4: "🦈",
    #     5 : "🐷",
    #     6: "🏠",
    #     7: "📷",
    # }
    # randomly permute classes
    # entities = np.random.permutation(entities).tolist()

    clip_lens.compute_text_cache(
        entities=entities,
        # prompt_template="a photo{}",
        emoji_mapping=emoji_mapping,
    )
    patch = torch.load("patch.pt")
    image = Image.open("notebooks/cat_dog.jpeg")
    clip_lens.compute_image_cache(image, patch=patch)
    # clip_lens.compute_image_cache(image)
    clip_lens.print_preds(layer_idx=-1)
    clip_lens.print_preds(layer_idx=0)
    clip_lens.display_patch_logit_lens(no_bias=False)
    # clip_lens.display_logit_lens(layer_idx=-1)
    print("hi")
