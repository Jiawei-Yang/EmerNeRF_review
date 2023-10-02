import logging
from typing import Callable, Dict, List, Literal, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch import Tensor

from radiance_fields.encodings import (
    HashEncoder,
    SHEncoder,
    SinusoidalEncoder,
    XYZ_Encoder,
    build_xyz_encoder_from_cfg,
)
from radiance_fields.nerf_utils import contract, find_topk_nearby_timestamps, trunc_exp

logger = logging.getLogger()


class MLP(nn.Module):
    """A simple MLP with skip connections."""

    def __init__(
        self,
        in_dims: int,
        out_dims: int,
        num_layers: int = 3,
        hidden_dims: Optional[int] = 256,
        skip_connections: Optional[Tuple[int]] = [0],
    ) -> None:
        super().__init__()
        self.in_dims = in_dims
        self.hidden_dims = hidden_dims
        self.n_output_dims = out_dims
        self.num_layers = num_layers
        self.skip_connections = skip_connections
        layers = []
        if self.num_layers == 1:
            layers.append(nn.Linear(in_dims, out_dims))
        else:
            for i in range(self.num_layers - 1):
                if i == 0:
                    layers.append(nn.Linear(in_dims, hidden_dims))
                elif i in skip_connections:
                    layers.append(nn.Linear(in_dims + hidden_dims, hidden_dims))
                else:
                    layers.append(nn.Linear(hidden_dims, hidden_dims))
            layers.append(nn.Linear(hidden_dims, out_dims))
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input = x
        for i, layer in enumerate(self.layers):
            if i in self.skip_connections:
                x = torch.cat([x, input], -1)
            x = layer(x)
            if i < len(self.layers) - 1:
                x = F.relu(x)
        return x


class RadianceField(nn.Module):
    def __init__(
        self,
        aabb: Union[Tensor, List[float]],
        xyz_encoder: HashEncoder,
        dynamic_xyz_encoder: Optional[HashEncoder] = None,
        flow_xyz_encoder: Optional[HashEncoder] = None,
        num_dims: int = 3,
        density_activation: Callable = lambda x: trunc_exp(x - 1),
        unbounded: bool = False,
        geometry_feature_dim: int = 15,
        base_mlp_layer_width: int = 64,
        head_mlp_layer_width: int = 64,
        enable_cam_embedding: bool = False,
        enable_img_embedding: bool = False,
        num_cams: int = 3,
        appearance_embedding_dim: int = 16,
        semantic_feature_dim: int = 64,
        dino_mlp_layer_width: int = 256,
        dino_embedding_dim: int = 384,
        enable_sky_head: bool = False,
        enable_dino_head: bool = False,
        enable_shadow_head: bool = False,
        num_train_timestamps: int = 0,
        interpolate_xyz_encoding: bool = False,
        enable_learnable_pe: bool = False,
        enable_temporal_interpolation: bool = False,
    ) -> None:
        super().__init__()
        # scene properties
        if not isinstance(aabb, Tensor):
            aabb = torch.tensor(aabb, dtype=torch.float32)
        self.register_buffer("aabb", aabb)
        self.unbounded = unbounded
        self.num_cams = num_cams
        self.num_dims = num_dims
        self.density_activation = density_activation

        # appearance embedding
        self.enable_cam_embedding = enable_cam_embedding
        self.enable_img_embedding = enable_img_embedding
        self.appearance_embedding_dim = appearance_embedding_dim

        self.geometry_feature_dim = geometry_feature_dim
        self.interpolate_xyz_encoding = interpolate_xyz_encoding

        if not enable_dino_head:
            semantic_feature_dim = 0
        self.semantic_feature_dim = semantic_feature_dim

        # ======== Static Field ======== #
        self.xyz_encoder = xyz_encoder
        self.base_mlp = nn.Sequential(
            nn.Linear(self.xyz_encoder.n_output_dims, base_mlp_layer_width),
            nn.ReLU(),
            nn.Linear(
                base_mlp_layer_width, geometry_feature_dim + semantic_feature_dim
            ),
        )

        # ======== Dynamic Field ======== #
        self.dynamic_xyz_encoder = dynamic_xyz_encoder
        self.enable_temporal_interpolation = enable_temporal_interpolation
        if self.dynamic_xyz_encoder is not None:
            self.register_buffer(
                "training_timestamps", torch.zeros(num_train_timestamps)
            )
            self.dynamic_base_mlp = nn.Sequential(
                nn.Linear(self.dynamic_xyz_encoder.n_output_dims, base_mlp_layer_width),
                nn.ReLU(),
                nn.Linear(
                    base_mlp_layer_width,
                    geometry_feature_dim + semantic_feature_dim,
                ),
            )
            self.dynamic_features_per_level = (
                self.dynamic_xyz_encoder.n_features_per_level
            )

        # ======== Flow Field ======== #
        self.flow_xyz_encoder = flow_xyz_encoder
        if self.flow_xyz_encoder is not None:
            self.flow_mlp = nn.Sequential(
                nn.Linear(
                    self.flow_xyz_encoder.n_output_dims,
                    base_mlp_layer_width,
                ),
                nn.ReLU(),
                nn.Linear(base_mlp_layer_width, base_mlp_layer_width),
                nn.ReLU(),
                nn.Linear(base_mlp_layer_width, 6),
            )

        # appearance embedding
        if self.enable_cam_embedding:
            self.appearance_embedding = nn.Embedding(num_cams, appearance_embedding_dim)
        elif self.enable_img_embedding:
            self.appearance_embedding = nn.Embedding(
                num_train_timestamps * num_cams, appearance_embedding_dim
            )
        else:
            self.appearance_embedding = None
        # directions
        self.direction_encoding = SinusoidalEncoder(
            n_input_dims=3, min_deg=0, max_deg=4
        )

        # ======== Color Head ======== #
        self.rgb_head = MLP(
            in_dims=geometry_feature_dim
            + self.direction_encoding.n_output_dims
            + (
                appearance_embedding_dim
                if self.enable_cam_embedding or self.enable_img_embedding
                else 2
            ),
            out_dims=3,
            num_layers=3,
            hidden_dims=head_mlp_layer_width,
            skip_connections=[1],
        )

        # ======== Shadow Head ======== #
        self.enable_shadow_head = enable_shadow_head
        if self.enable_shadow_head:
            self.shadow_head = nn.Sequential(
                nn.Linear(geometry_feature_dim, base_mlp_layer_width),
                nn.ReLU(),
                nn.Linear(base_mlp_layer_width, 1),
                nn.Sigmoid(),
            )

        # ======== Sky Head ======== #
        self.enable_sky_head = enable_sky_head
        if self.enable_sky_head:
            self.sky_head = MLP(
                in_dims=self.direction_encoding.n_output_dims
                + (
                    appearance_embedding_dim
                    if self.enable_cam_embedding or self.enable_img_embedding
                    else 0
                ),
                out_dims=3,
                num_layers=3,
                hidden_dims=head_mlp_layer_width,
                skip_connections=[1],
            )
            if enable_dino_head:
                # feature sky head
                self.dino_sky_head = nn.Sequential(
                    # TODO: remove appearance embedding from dino sky head
                    nn.Linear(
                        self.direction_encoding.n_output_dims
                        + (
                            appearance_embedding_dim
                            if self.enable_cam_embedding or self.enable_img_embedding
                            else 0
                        ),
                        dino_mlp_layer_width,
                    ),
                    nn.ReLU(),
                    nn.Linear(dino_mlp_layer_width, dino_mlp_layer_width),
                    nn.ReLU(),
                    nn.Linear(dino_mlp_layer_width, dino_embedding_dim),
                )

        # ======== Feature Head ======== #
        self.enable_dino_head = enable_dino_head
        if self.enable_dino_head:
            self.dino_head = nn.Sequential(
                nn.Linear(semantic_feature_dim, dino_mlp_layer_width),
                nn.ReLU(),
                nn.Linear(dino_mlp_layer_width, dino_mlp_layer_width),
                nn.ReLU(),
                nn.Linear(dino_mlp_layer_width, dino_embedding_dim),
            )
            # placeholder for dino_feats_reduction_mat, will be filled after
            # the dataset is loaded
            self.register_buffer(
                "dino_feats_reduction_mat", torch.zeros(dino_embedding_dim, 3)
            )
            self.register_buffer("color_norm_min", torch.zeros(3, dtype=torch.float32))
            self.register_buffer("color_norm_max", torch.ones(3, dtype=torch.float32))

            # positional embedding decomposition
            self.enable_learnable_pe = enable_learnable_pe
            if self.enable_learnable_pe:
                # globally-shared low-resolution learnable positional embedding map
                self.learnable_pe_map = nn.Parameter(
                    0.05 * torch.randn(1, dino_embedding_dim // 2, 80, 120),
                    requires_grad=True,
                )
                # positional embedding head to decode the PE map
                self.pe_head = nn.Sequential(
                    nn.Linear(dino_embedding_dim // 2, dino_embedding_dim),
                )

    def register_training_timestamps(self, timestamps: Tensor) -> None:
        if self.dynamic_xyz_encoder is not None:
            # register timestamps for time interpolation
            self.training_timestamps.copy_(timestamps)
            self.training_timestamps = self.training_timestamps.to(self.device)
            if len(self.training_timestamps) > 1:
                self.time_diff = (
                    self.training_timestamps[1] - self.training_timestamps[0]
                )
            else:
                self.time_diff = 0

    def set_aabb(self, aabb: Union[Tensor, List[float]]) -> None:
        if not isinstance(aabb, Tensor):
            aabb = torch.tensor(aabb, dtype=torch.float32)
        logger.info(f"Set aabb from {self.aabb} to {aabb}")
        self.aabb.copy_(aabb)
        self.aabb = self.aabb.to(self.device)

    def register_dino_feats_reduction_mat(
        self,
        dino_feats_reduction_mat: Tensor,
        color_norm_min: Tensor,
        color_norm_max: Tensor,
    ) -> None:
        # for visualization
        self.dino_feats_reduction_mat.copy_(dino_feats_reduction_mat)
        self.color_norm_min.copy_(color_norm_min)
        self.color_norm_max.copy_(color_norm_max)
        self.dino_feats_reduction_mat = self.dino_feats_reduction_mat.to(self.device)
        self.color_norm_min = self.color_norm_min.to(self.device)
        self.color_norm_max = self.color_norm_max.to(self.device)

    @property
    def device(self) -> torch.device:
        return self.aabb.device

    def contract_points(
        self,
        positions: Tensor,
    ) -> Tensor:
        if self.unbounded:
            # use infinte norm to contract the positions for cuboid aabb
            normed_positions = contract(positions, self.aabb, ord=float("inf"))
        else:
            aabb_min, aabb_max = torch.split(self.aabb, 3, dim=-1)
            normed_positions = (positions - aabb_min) / (aabb_max - aabb_min)
        selector = (
            ((normed_positions > 0.0) & (normed_positions < 1.0))
            .all(dim=-1)
            .to(positions)
        )
        normed_positions = normed_positions * selector.unsqueeze(-1)
        return normed_positions

    def forward_static_hash(
        self,
        positions: Tensor,
    ) -> Tensor:
        normed_positions = self.contract_points(positions)
        xyz_encoding = self.xyz_encoder(normed_positions.view(-1, self.num_dims))
        encoded_features = self.base_mlp(xyz_encoding).view(
            list(normed_positions.shape[:-1]) + [-1]
        )
        return encoded_features, normed_positions

    def forward_dynamic_hash(
        self,
        normed_positions: Tensor,
        timestamp: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        # To be fixed.
        # if self.training or not self.enable_temporal_interpolation:
        if True:
            temporal_positions = torch.cat([normed_positions, timestamp], dim=-1)
            dynamic_xyz_encoding = self.dynamic_xyz_encoder(
                temporal_positions.view(-1, self.num_dims + 1)
            ).view(list(temporal_positions.shape[:-1]) + [-1])
            encoded_dynamic_feats = self.dynamic_base_mlp(dynamic_xyz_encoding)
        else:
            encoded_dynamic_feats = temporal_interpolation(
                timestamp,
                self.training_timestamps,
                normed_positions,
                self.dynamic_xyz_encoder,
                self.dynamic_base_mlp,
                interpolate_xyz_encoding=self.interpolate_xyz_encoding,
            )
        return encoded_dynamic_feats

    def forward_flow_hash(
        self,
        normed_positions: Tensor,
        timestamp: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        if self.training or not self.enable_temporal_interpolation:
            temporal_positions = torch.cat([normed_positions, timestamp], dim=-1)
            flow_xyz_encoding = self.flow_xyz_encoder(
                temporal_positions.view(-1, self.num_dims + 1)
            ).view(list(temporal_positions.shape[:-1]) + [-1])
            flow = self.flow_mlp(flow_xyz_encoding)
        else:
            flow = temporal_interpolation(
                timestamp,
                self.training_timestamps,
                normed_positions,
                self.flow_xyz_encoder,
                self.flow_mlp,
                interpolate_xyz_encoding=True,
            )
        return flow

    def forward(
        self,
        positions: Tensor,
        directions: Tensor = None,
        data_dict: Dict[str, Tensor] = None,
        return_density_only: bool = False,
    ) -> Dict[str, Tensor]:
        results_dict = {}
        encoded_features, normed_positions = self.forward_static_hash(positions)
        geo_feats, semantic_feats = torch.split(
            encoded_features,
            [self.geometry_feature_dim, self.semantic_feature_dim],
            dim=-1,
        )
        static_density = self.density_activation(geo_feats[..., 0])
        if self.dynamic_xyz_encoder is not None:
            if "timestamp" in data_dict:
                timestamp = data_dict["timestamp"].unsqueeze(-1)
            elif "lidar_timestamp" in data_dict:
                timestamp = data_dict["lidar_timestamp"].unsqueeze(-1)
            else:
                raise NotImplementedError("Timestamp is not provided.")
            dynamic_feats = self.forward_dynamic_hash(normed_positions, timestamp)
            if self.flow_xyz_encoder is not None:
                flow = self.forward_flow_hash(normed_positions, timestamp)
                forward_flow = flow[..., :3]
                backward_flow = flow[..., 3:]
                results_dict["forward_flow"] = forward_flow
                results_dict["backward_flow"] = backward_flow
                temporal_aggregation_results = self.temporal_aggregation(
                    positions, timestamp, forward_flow, backward_flow, dynamic_feats
                )
                dynamic_feats = temporal_aggregation_results["dynamic_feats"]
                results_dict[
                    "forward_pred_backward_flow"
                ] = temporal_aggregation_results["forward_pred_backward_flow"]
                results_dict[
                    "backward_pred_forward_flow"
                ] = temporal_aggregation_results["backward_pred_forward_flow"]

            (
                dynamic_geo_feats,
                dynamic_semantic_feats,
            ) = torch.split(
                dynamic_feats,
                [self.geometry_feature_dim, self.semantic_feature_dim],
                dim=-1,
            )
            dynamic_density = self.density_activation(dynamic_geo_feats[..., 0])
            density = static_density + dynamic_density
            results_dict.update(
                {
                    "density": density,
                    "static_density": static_density,
                    "dynamic_density": dynamic_density,
                }
            )
            if return_density_only:
                return results_dict
            rgb_results = self.query_rgb(
                directions, geo_feats, dynamic_geo_feats, data_dict=data_dict
            )
            results_dict["dynamic_rgb"] = rgb_results["dynamic_rgb"]
            results_dict["static_rgb"] = rgb_results["rgb"]
            if self.enable_shadow_head:
                shadow_ratio = self.shadow_head(dynamic_geo_feats)
                results_dict["shadow_ratio"] = shadow_ratio
        else:
            results_dict["density"] = static_density
            if return_density_only:
                return results_dict
            rgb_results = self.query_rgb(directions, geo_feats, data_dict=data_dict)
            results_dict["rgb"] = rgb_results["rgb"]

        if self.enable_dino_head:
            # query on demand
            if self.enable_learnable_pe:
                learnable_pe_map = (
                    F.grid_sample(
                        self.learnable_pe_map,
                        data_dict["pix_coords"].reshape(1, 1, -1, 2) * 2 - 1,
                        align_corners=False,
                        mode="bilinear",
                    )
                    .squeeze(2)
                    .squeeze(0)
                    .permute(1, 0)
                )
                dino_pe = self.pe_head(learnable_pe_map)
                results_dict["dino_pe"] = dino_pe
            dino_feats = self.dino_head(semantic_feats)

            if self.dynamic_xyz_encoder is not None:
                dynamic_dino_feats = self.dino_head(dynamic_semantic_feats)
                results_dict["static_dino_feat"] = dino_feats
                results_dict["dynamic_dino_feat"] = dynamic_dino_feats
            else:
                results_dict["dino_feat"] = dino_feats
        return results_dict

    def temporal_aggregation(
        self,
        positions: Tensor,
        timestamp: Tensor,
        forward_flow: Tensor,
        backward_flow: Tensor,
        dynamic_feats: Tensor,
    ) -> Tensor:
        if self.training:
            noise = torch.rand_like(forward_flow)[..., 0:1]
        else:
            noise = torch.ones_like(forward_flow)[..., 0:1]
        forward_warped_positions = self.contract_points(
            positions + forward_flow * noise
        )
        backward_warped_positions = self.contract_points(
            positions + backward_flow * noise
        )
        forward_warped_time = torch.clamp(timestamp + self.time_diff * noise, 0, 1.0)
        backward_warped_time = torch.clamp(timestamp - self.time_diff * noise, 0, 1.0)
        forward_dynamic_feats = self.forward_dynamic_hash(
            forward_warped_positions,
            forward_warped_time,
        )
        backward_dynamic_feats = self.forward_dynamic_hash(
            backward_warped_positions,
            backward_warped_time,
        )
        forward_pred_flow = self.forward_flow_hash(
            forward_warped_positions,
            forward_warped_time,
        )
        backward_pred_flow = self.forward_flow_hash(
            backward_warped_positions,
            backward_warped_time,
        )
        dynamic_feats = (
            dynamic_feats + 0.5 * forward_dynamic_feats + 0.5 * backward_dynamic_feats
        ) / 2.0
        return {
            "dynamic_feats": dynamic_feats,
            "forward_pred_backward_flow": forward_pred_flow[..., 3:],
            "backward_pred_forward_flow": backward_pred_flow[..., :3],
        }

    def query_rgb(
        self,
        directions: Tensor,
        geo_feats: Tensor,
        dynamic_geo_feats: Tensor = None,
        data_dict: Dict[str, Tensor] = None,
    ) -> Tensor:
        directions = (directions + 1.0) / 2.0
        h = self.direction_encoding(directions.reshape(-1, directions.shape[-1])).view(
            *directions.shape[:-1], -1
        )
        if self.enable_cam_embedding or self.enable_img_embedding:
            if "cam_idx" in data_dict and self.enable_cam_embedding:
                appearance_embedding = self.appearance_embedding(data_dict["cam_idx"])
            elif "img_idx" in data_dict and self.enable_img_embedding:
                appearance_embedding = self.appearance_embedding(data_dict["img_idx"])
            else:
                # use mean appearance embedding
                appearance_embedding = torch.ones(
                    (*directions.shape[:-1], self.appearance_embedding_dim),
                    device=directions.device,
                ) * self.appearance_embedding.weight.mean(dim=0)
            h = torch.cat([h, appearance_embedding], dim=-1)

        rgb = self.rgb_head(torch.cat([h, geo_feats], dim=-1))
        rgb = F.sigmoid(rgb)
        results = {"rgb": rgb}

        if self.dynamic_xyz_encoder is not None:
            assert (
                dynamic_geo_feats is not None
            ), "Dynamic geometry features are not provided."
            dynamic_rgb = self.rgb_head(torch.cat([h, dynamic_geo_feats], dim=-1))
            dynamic_rgb = F.sigmoid(dynamic_rgb)
            results["dynamic_rgb"] = dynamic_rgb
        return results

    def query_sky(
        self, directions: Tensor, data_dict: Dict[str, Tensor] = None
    ) -> Dict[str, Tensor]:
        if self.enable_sky_head:
            if len(directions.shape) == 2:
                dd = self.direction_encoding(directions).to(directions)
            else:
                dd = self.direction_encoding(directions[:, 0]).to(directions)
            if self.enable_cam_embedding or self.enable_img_embedding:
                if "cam_idx" in data_dict and self.enable_cam_embedding:
                    appearance_embedding = self.appearance_embedding(
                        data_dict["cam_idx"]
                    )
                elif "img_idx" in data_dict and self.enable_img_embedding:
                    appearance_embedding = self.appearance_embedding(
                        data_dict["img_idx"]
                    )
                else:
                    # use mean appearance embedding
                    appearance_embedding = torch.ones(
                        (*directions.shape[:-1], self.appearance_embedding_dim),
                        device=directions.device,
                    ) * self.appearance_embedding.weight.mean(dim=0)
                dd = torch.cat([dd, appearance_embedding], dim=-1)
            rgb_sky = self.sky_head(dd).to(directions)
            rgb_sky = F.sigmoid(rgb_sky)
            results = {"rgb_sky": rgb_sky}
            if self.enable_dino_head:
                self.dino_sky_head(dd).to(directions)
                results["dino_sky_feat"] = self.dino_sky_head(dd).to(directions)
            return results
        else:
            return {}

    def query_flow(
        self, positions: Tensor, timestamps: Tensor, query_density: bool = True
    ) -> Dict[str, Tensor]:
        normed_positions = self.contract_points(positions)
        timestamps = timestamps.unsqueeze(-1)
        if self.training or not self.enable_temporal_interpolation:
            temporal_positions = torch.cat([normed_positions, timestamps], dim=-1)
            flow_xyz_encoding = self.flow_xyz_encoder(temporal_positions.view(-1, 4))
            flow = self.flow_mlp(flow_xyz_encoding).view(
                list(positions.shape[:-1]) + [-1]
            )
        else:
            if len(timestamps.shape) == 2:
                timestamp_slice = timestamps[:, 0]
            else:
                timestamp_slice = timestamps[:, 0, 0]
            closest_timestamps = find_topk_nearby_timestamps(
                self.training_timestamps, timestamp_slice
            )
            if torch.allclose(closest_timestamps[:, 0], timestamp_slice):
                temporal_positions = torch.cat([normed_positions, timestamps], dim=-1)
                flow_xyz_encoding = self.flow_xyz_encoder(
                    temporal_positions.view(-1, self.num_dims + 1)
                ).view(list(temporal_positions.shape[:-1]) + [-1])
                flow = self.flow_mlp(flow_xyz_encoding)
            else:
                left_timestamps, right_timestamps = (
                    closest_timestamps[:, 0],
                    closest_timestamps[:, 1],
                )
                left_timestamps = left_timestamps.unsqueeze(-1).repeat(
                    1, normed_positions.shape[1]
                )
                right_timestamps = right_timestamps.unsqueeze(-1).repeat(
                    1, normed_positions.shape[1]
                )
                left_temporal_positions = torch.cat(
                    [normed_positions, left_timestamps.unsqueeze(-1)], dim=-1
                )
                right_temporal_positions = torch.cat(
                    [normed_positions, right_timestamps.unsqueeze(-1)], dim=-1
                )
                offset = (
                    (
                        (timestamp_slice - left_timestamps[:, 0])
                        / (right_timestamps[:, 0] - left_timestamps[:, 0])
                    )
                    .unsqueeze(-1)
                    .unsqueeze(-1)
                )
                left_flow_xyz_encoding = self.flow_xyz_encoder(
                    left_temporal_positions.view(-1, self.num_dims + 1)
                ).view(list(left_temporal_positions.shape[:-1]) + [-1])
                right_flow_xyz_encoding = self.flow_xyz_encoder(
                    right_temporal_positions.view(-1, self.num_dims + 1)
                ).view(list(right_temporal_positions.shape[:-1]) + [-1])
                # method 1:
                if self.interpolate_xyz_encoding:
                    flow_xyz_encoding = (
                        left_flow_xyz_encoding * (1 - offset)
                        + right_flow_xyz_encoding * offset
                    )
                    flow = self.flow_mlp(flow_xyz_encoding)
                else:
                    raise NotImplementedError
                # method 2:
                # left_flow = self.flow_mlp(left_flow_xyz_encoding).view(
                #         list(left_temporal_positions.shape[:-1]) + [-1]
                #     )
                # right_flow = self.flow_mlp(right_flow_xyz_encoding).view(
                #         list(right_temporal_positions.shape[:-1]) + [-1]
                # )
                # flow = left_flow * (1 - offset) + right_flow * offset

        results = {
            "forward_flow": flow[..., :3],
            "backward_flow": flow[..., 3:],
        }
        if query_density:
            dynamic_feats, dynamic_xyz_encoding, _ = self.forward_dynamic_hash(
                normed_positions, timestamps
            )
            (
                dynamic_geo_feats,
                dynamic_semantic_feats,
            ) = torch.split(
                dynamic_feats,
                [self.geometry_feature_dim, self.semantic_feature_dim],
                dim=-1,
            )
            dynamic_density = self.density_activation(dynamic_geo_feats[..., 0])
            results["dynamic_density"] = dynamic_density.detach()
        return results

    def query_attributes(
        self,
        positions: Tensor,
        timestamps: Tensor = None,
        query_dino_feats: bool = True,
    ):
        results_dict = {}
        encoded_features, normed_positions = self.forward_static_hash(positions)
        geo_feats, semantic_feats = torch.split(
            encoded_features,
            [self.geometry_feature_dim, self.semantic_feature_dim],
            dim=-1,
        )
        static_density = self.density_activation(geo_feats[..., 0])
        if self.dynamic_xyz_encoder is not None:
            dynamic_feats = self.forward_dynamic_hash(normed_positions, timestamps)
            if self.flow_xyz_encoder is not None:
                flow = self.forward_flow_hash(normed_positions, timestamps)
                forward_flow = flow[..., :3]
                backward_flow = flow[..., 3:]
                results_dict["forward_flow"] = forward_flow
                results_dict["backward_flow"] = backward_flow
                temporal_aggregation_results = self.temporal_aggregation(
                    positions, timestamps, forward_flow, backward_flow, dynamic_feats
                )
                dynamic_feats = temporal_aggregation_results["dynamic_feats"]
                results_dict[
                    "forward_pred_backward_flow"
                ] = temporal_aggregation_results["forward_pred_backward_flow"]
                results_dict[
                    "backward_pred_forward_flow"
                ] = temporal_aggregation_results["backward_pred_forward_flow"]

            (
                dynamic_geo_feats,
                dynamic_semantic_feats,
            ) = torch.split(
                dynamic_feats,
                [self.geometry_feature_dim, self.semantic_feature_dim],
                dim=-1,
            )
            dynamic_density = self.density_activation(dynamic_geo_feats[..., 0])
            density = static_density + dynamic_density
            results_dict.update(
                {
                    "density": density,
                    "static_density": static_density,
                    "dynamic_density": dynamic_density,
                    # "occupancy": occupancy,
                }
            )
        else:
            results_dict["density"] = static_density
        if self.enable_dino_head and query_dino_feats:
            # query on demand
            dino_feats = self.dino_head(semantic_feats)
            if self.dynamic_xyz_encoder is not None:
                dynamic_dino_feats = self.dino_head(dynamic_semantic_feats)
                results_dict["static_dino_feat"] = dino_feats
                results_dict["dynamic_dino_feat"] = dynamic_dino_feats
                results_dict["dino_feat"] = (
                    static_density.unsqueeze(-1) * dino_feats
                    + dynamic_density.unsqueeze(-1) * dynamic_dino_feats
                ) / (density.unsqueeze(-1) + 1e-6)
            else:
                results_dict["dino_feat"] = dino_feats
        return results_dict


class DensityField(nn.Module):
    def __init__(
        self,
        xyz_encoder: XYZ_Encoder,
        aabb: Union[Tensor, List[float]],
        dynamic_xyz_encoder: Optional[XYZ_Encoder] = None,
        num_dims: int = 3,
        density_activation: Callable = lambda x: trunc_exp(x - 1),
        unbounded: bool = False,
        base_mlp_layer_width: int = 64,
        num_train_timestamps: int = 0,
        enable_dynamic_branch: bool = False,
        interpolate_xyz_encoding: bool = False,
    ) -> None:
        super().__init__()
        if not isinstance(aabb, Tensor):
            aabb = torch.tensor(aabb, dtype=torch.float32)
        self.register_buffer("aabb", aabb)
        self.num_dims = num_dims
        self.density_activation = density_activation
        self.unbounded = unbounded
        self.xyz_encoder = xyz_encoder
        self.dynamic_xyz_encoder = dynamic_xyz_encoder
        self.enable_dynamic_branch = enable_dynamic_branch
        self.interpolate_xyz_encoding = interpolate_xyz_encoding

        # density head
        self.base_mlp = nn.Sequential(
            nn.Linear(self.xyz_encoder.n_output_dims, base_mlp_layer_width),
            nn.ReLU(),
            nn.Linear(base_mlp_layer_width, 1),
        )
        if enable_dynamic_branch:
            self.register_buffer(
                "training_timestamps", torch.zeros(num_train_timestamps)
            )
            self.dynamic_base_mlp = nn.Sequential(
                nn.Linear(self.dynamic_xyz_encoder.n_output_dims, base_mlp_layer_width),
                nn.ReLU(),
                nn.Linear(base_mlp_layer_width, 1),
            )

    @property
    def device(self) -> torch.device:
        return self.aabb.device

    def register_training_timestamps(self, timestamps: Tensor) -> None:
        if self.dynamic_xyz_encoder is not None:
            self.training_timestamps.copy_(timestamps)
            self.training_timestamps = self.training_timestamps.to(self.device)

    def set_aabb(self, aabb: Union[Tensor, List[float]]) -> None:
        if not isinstance(aabb, Tensor):
            aabb = torch.tensor(aabb, dtype=torch.float32)
        logger.info(f"Set propnet aabb from {self.aabb} to {aabb}")
        self.aabb.copy_(aabb)
        self.aabb = self.aabb.to(self.device)

    def forward_dynamic_hash(
        self, normed_positions: Tensor, timestamp: Tensor
    ) -> Tensor:
        if self.training:
            temporal_positions = torch.cat([normed_positions, timestamp], dim=-1)
            dynamic_xyz_encoding = self.dynamic_xyz_encoder(
                temporal_positions.view(-1, self.num_dims + 1)
            ).view(list(temporal_positions.shape[:-1]) + [-1])
            density_before_activation = self.dynamic_base_mlp(dynamic_xyz_encoding)
        else:
            density_before_activation = temporal_interpolation(
                timestamp,
                self.training_timestamps,
                normed_positions,
                self.dynamic_xyz_encoder,
                self.dynamic_base_mlp,
                interpolate_xyz_encoding=self.interpolate_xyz_encoding,
            )
        density = self.density_activation(density_before_activation)
        return density

    def forward(
        self,
        positions: Tensor,
        data_dict: Dict[str, Tensor] = None,
    ) -> Dict[str, Tensor]:
        if self.unbounded:
            # use infinte norm to contract the positions for cuboid aabb
            positions = contract(positions, self.aabb, ord=float("inf"))
        else:
            aabb_min, aabb_max = torch.split(self.aabb, 3, dim=-1)
            positions = (positions - aabb_min) / (aabb_max - aabb_min)
        selector = ((positions > 0.0) & (positions < 1.0)).all(dim=-1).to(positions)
        positions = positions * selector.unsqueeze(-1)
        xyz_encoding = self.xyz_encoder(positions.view(-1, self.num_dims))
        density_before_activation = self.base_mlp(xyz_encoding).view(
            list(positions.shape[:-1]) + [-1]
        )
        static_density = self.density_activation(density_before_activation)
        if self.dynamic_xyz_encoder is not None:
            assert (
                "timestamp" in data_dict or "lidar_timestamp" in data_dict
            ), "timestamp must be provided"
            if "timestamp" in data_dict:
                timestamp = data_dict["timestamp"].unsqueeze(-1)
            else:
                timestamp = data_dict["lidar_timestamp"].unsqueeze(-1)
            dynamic_density = self.forward_dynamic_hash(positions, timestamp)
            density = static_density + dynamic_density
            results_dict = {
                "density": density,
                "static_density": static_density,
                "dynamic_density": dynamic_density,
            }
        else:
            results_dict = {"density": static_density}
        return results_dict


def temporal_interpolation(
    timestamp: Tensor,
    training_timestamps: Tensor,
    normed_positions: Tensor,
    hash_encoder: HashEncoder,
    mlp: nn.Module,
    interpolate_xyz_encoding: bool = False,
) -> Tensor:
    if len(timestamp.shape) == 2:
        timestamp_slice = timestamp[:, 0]
    else:
        timestamp_slice = timestamp[:, 0, 0]
    closest_timestamps = find_topk_nearby_timestamps(
        training_timestamps, timestamp_slice
    )
    if torch.allclose(closest_timestamps[:, 0], timestamp_slice):
        temporal_positions = torch.cat([normed_positions, timestamp], dim=-1)
        xyz_encoding = hash_encoder(
            temporal_positions.view(-1, temporal_positions.shape[-1])
        ).view(list(temporal_positions.shape[:-1]) + [-1])
        encoded_feats = mlp(xyz_encoding)
    else:
        left_timestamps, right_timestamps = (
            closest_timestamps[:, 0],
            closest_timestamps[:, 1],
        )
        left_timestamps = left_timestamps.unsqueeze(-1).repeat(
            1, normed_positions.shape[1]
        )
        right_timestamps = right_timestamps.unsqueeze(-1).repeat(
            1, normed_positions.shape[1]
        )
        left_temporal_positions = torch.cat(
            [normed_positions, left_timestamps.unsqueeze(-1)], dim=-1
        )
        right_temporal_positions = torch.cat(
            [normed_positions, right_timestamps.unsqueeze(-1)], dim=-1
        )
        offset = (
            (
                (timestamp_slice - left_timestamps[:, 0])
                / (right_timestamps[:, 0] - left_timestamps[:, 0])
            )
            .unsqueeze(-1)
            .unsqueeze(-1)
        )
        left_xyz_encoding = hash_encoder(
            left_temporal_positions.view(-1, left_temporal_positions.shape[-1])
        ).view(list(left_temporal_positions.shape[:-1]) + [-1])
        right_xyz_encoding = hash_encoder(
            right_temporal_positions.view(-1, left_temporal_positions.shape[-1])
        ).view(list(right_temporal_positions.shape[:-1]) + [-1])
        if interpolate_xyz_encoding:
            encoded_feats = mlp(
                left_xyz_encoding * (1 - offset) + right_xyz_encoding * offset
            )
        else:
            encoded_feats = (
                mlp(left_xyz_encoding) * (1 - offset) + mlp(right_xyz_encoding) * offset
            )

    return encoded_feats


def build_radiance_field_from_cfg(cfg, verbose=True) -> RadianceField:
    xyz_encoder = build_xyz_encoder_from_cfg(cfg.xyz_encoder, verbose=verbose)
    dynamic_xyz_encoder = None
    flow_xyz_encoder = None
    if cfg.head.enable_dynamic_branch:
        dynamic_xyz_encoder = build_xyz_encoder_from_cfg(
            cfg.dynamic_xyz_encoder, verbose=verbose
        )
    if cfg.head.enable_flow_branch:
        flow_xyz_encoder = HashEncoder(
            n_input_dims=4,
            n_levels=10,
            base_resolution=16,
            max_resolution=4096,
            log2_hashmap_size=18,
            n_features_per_level=4,
        )
    return RadianceField(
        aabb=cfg.scene.aabb,
        xyz_encoder=xyz_encoder,
        dynamic_xyz_encoder=dynamic_xyz_encoder,
        flow_xyz_encoder=flow_xyz_encoder,
        unbounded=cfg.scene.unbounded,
        num_cams=cfg.num_cams,
        geometry_feature_dim=cfg.neck.geometry_feature_dim,
        base_mlp_layer_width=cfg.neck.base_mlp_layer_width,
        head_mlp_layer_width=cfg.head.head_mlp_layer_width,
        enable_cam_embedding=cfg.head.enable_cam_embedding,
        enable_img_embedding=cfg.head.enable_img_embedding,
        appearance_embedding_dim=cfg.head.appearance_embedding_dim,
        enable_sky_head=cfg.head.enable_sky_head,
        enable_dino_head=cfg.head.enable_dino_head,
        semantic_feature_dim=cfg.neck.semantic_feature_dim,
        dino_mlp_layer_width=cfg.head.dino_mlp_layer_width,
        dino_embedding_dim=cfg.head.dino_embedding_dim,
        enable_shadow_head=cfg.head.enable_shadow_head,
        num_train_timestamps=cfg.num_train_timestamps,  # placeholder
        interpolate_xyz_encoding=cfg.head.interpolate_xyz_encoding,
        enable_learnable_pe=cfg.head.enable_learnable_pe,
        enable_temporal_interpolation=cfg.head.enable_temporal_interpolation,
    )


def build_density_field(
    aabb: Union[Tensor, List[float]],
    type: Literal["HashEncoder"] = "HashEncoder",
    n_input_dims: int = 3,
    n_levels: int = 5,
    base_resolution: int = 16,
    max_resolution: int = 128,
    log2_hashmap_size: int = 20,
    n_features_per_level: int = 2,
    unbounded: bool = True,
    enable_temporal_propnet: bool = False,
    nerf_model_cfg: Optional[OmegaConf] = None,
) -> DensityField:
    if type == "HashEncoder":
        xyz_encoder = HashEncoder(
            n_input_dims=n_input_dims,
            n_levels=n_levels,
            base_resolution=base_resolution,
            max_resolution=max_resolution,
            log2_hashmap_size=log2_hashmap_size,
            n_features_per_level=n_features_per_level,
        )
    else:
        raise NotImplementedError(f"Unknown (xyz_encoder) type: {type}")
    if enable_temporal_propnet:
        dynamic_xyz_encoder = HashEncoder(
            n_input_dims=n_input_dims + 1,
            n_levels=n_levels,
            base_resolution=base_resolution,
            max_resolution=max_resolution,
            log2_hashmap_size=log2_hashmap_size,
            n_features_per_level=n_features_per_level,
        )
    else:
        dynamic_xyz_encoder = None
    return DensityField(
        xyz_encoder=xyz_encoder,
        aabb=aabb,
        dynamic_xyz_encoder=dynamic_xyz_encoder,
        unbounded=unbounded,
        num_train_timestamps=nerf_model_cfg.num_train_timestamps,
        interpolate_xyz_encoding=True,
    )
