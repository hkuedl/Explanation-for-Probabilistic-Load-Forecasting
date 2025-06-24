import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import *
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error as MSE
from torch.distributions.normal import Normal
import warnings
import os
warnings.filterwarnings("ignore")
from layers import (
    ConditionalFeatureMixing,
    ConditionalMixerLayer,
    TimeBatchNorm2d,
    feature_to_time,
    time_to_feature,
)
from captum.attr import (
    GradientShap,
    DeepLift,
    DeepLiftShap,
    IntegratedGradients,
    LayerConductance,
    NeuronConductance,
    NoiseTunnel,
)

import argparse
def mysigma(sigma):
    return torch.mean(sigma)

class SparseAutoEncoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(SparseAutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size)
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, input_size)
        )

    def forward(self, x):
        h = self.encoder(x)
        x = self.decoder(h)
        return x

class TSMixerExt_SAE(nn.Module):
    """TSMixer model for time series forecasting.

    This model forecasts time series data by integrating historical time series data,
    future known inputs, and static contextual information. It uses a combination of
    conditional feature mixing and mixer layers to process and combine these different
    types of data for effective forecasting.

    Args:
        sequence_length: The length of the input time series sequences.
        prediction_length: The length of the output prediction sequences.
        activation_fn: The name of the activation function to be used.
        num_blocks: The number of mixer blocks in the model.
        dropout_rate: The dropout rate used in the mixer layers.
        input_channels: The number of channels in the historical time series data.
        extra_channels: The number of channels in the extra (future known) inputs.
        hidden_channels: The number of hidden channels used in the mixer layers.
        static_channels: The number of channels in the static feature inputs.
        ff_dim: The inner dimension of the feedforward network in the mixer layers.
        output_channels: The number of output channels for the final output. If None,
                         defaults to the number of input_channels.
        normalize_before: Whether to apply layer normalization before or after mixer layer.
        norm_type: The type of normalization to use. "batch" or "layer".
    """

    def __init__(
        self,
        sequence_length: int,
        prediction_length: int,
        activation_fn: str = "relu",
        num_blocks: int = 2,
        dropout_rate: float = 0.1,
        input_channels: int = 1,
        extra_channels: int = 1,
        hidden_channels: int = 64,
        static_channels: int = 1,
        ff_dim: int = 64,
        output_channels: int = None,
        normalize_before: bool = False,
        norm_type: str = "layer",
    ):
        assert static_channels > 0, "static_channels must be greater than 0"
        super().__init__()

        # Transform activation_fn string to callable function
        if hasattr(F, activation_fn):
            activation_fn = getattr(F, activation_fn)
        else:
            raise ValueError(f"Unknown activation function: {activation_fn}")

        # Transform norm_type to callable
        assert norm_type in {
            "batch",
            "layer",
        }, f"Invalid norm_type: {norm_type}, must be one of batch, layer."
        norm_type = TimeBatchNorm2d if norm_type == "batch" else nn.LayerNorm

        self.fc_hist = nn.Linear(sequence_length, prediction_length)
        self.fc_out = nn.Linear(hidden_channels, output_channels or input_channels)
        self.fc_presigma = nn.Linear(hidden_channels, output_channels or input_channels)
        self.fc_sigma = nn.Softplus()

        self.feature_mixing_hist = ConditionalFeatureMixing(
            sequence_length=prediction_length,
            input_channels=input_channels + extra_channels,
            output_channels=hidden_channels,
            static_channels=static_channels,
            ff_dim=ff_dim,
            activation_fn=activation_fn,
            dropout_rate=dropout_rate,
            normalize_before=normalize_before,
            norm_type=norm_type,
        )
        self.feature_mixing_future = ConditionalFeatureMixing(
            sequence_length=prediction_length,
            input_channels=extra_channels,
            output_channels=hidden_channels,
            static_channels=static_channels,
            ff_dim=ff_dim,
            activation_fn=activation_fn,
            dropout_rate=dropout_rate,
            normalize_before=normalize_before,
            norm_type=norm_type,
        )

        self.conditional_mixer = self._build_mixer(
            num_blocks,
            hidden_channels,
            prediction_length,
            ff_dim=ff_dim,
            static_channels=static_channels,
            activation_fn=activation_fn,
            dropout_rate=dropout_rate,
            normalize_before=normalize_before,
            norm_type=norm_type,
        )
        self.sae_hist = SparseAutoEncoder(input_size=extra_channels, hidden_size=64)  # Assume hidden size is 64
        self.sae_future = SparseAutoEncoder(input_size=extra_channels, hidden_size=64)

    @staticmethod
    def _build_mixer(
        num_blocks: int, hidden_channels: int, prediction_length: int, **kwargs
    ):
        """Build the mixer blocks for the model."""
        channels = [2 * hidden_channels] + [hidden_channels] * (num_blocks - 1)

        return nn.ModuleList(
            [
                ConditionalMixerLayer(
                    input_channels=in_ch,
                    output_channels=out_ch,
                    sequence_length=prediction_length,
                    **kwargs,
                )
                for in_ch, out_ch in zip(channels[:-1], channels[1:])
            ]
        )

    def forward(
        self,
        x_hist: torch.Tensor,
        x_extra_hist: torch.Tensor,
        x_extra_future: torch.Tensor,
        x_static: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass for the TSMixer model.

        Processes historical and future data, along with static features, to produce a
        forecast.

        Args:
            x_hist: Historical time series data (batch_size, sequence_length,
                input_channels).
            x_extra_hist: Additional historical data (batch_size, sequence_length,
                extra_channels).
            x_extra_future: Future known data (batch_size, prediction_length,
                extra_channels).
            x_static: Static contextual data (batch_size, static_channels).

        Returns:
            The output tensor representing the forecast (batch_size, prediction_length,
            output_channels).
        """
        # x_extra_hist,hidden_hist = self.sae_hist(x_extra_hist)
        x_extra_future = x_extra_future+self.sae_future(x_extra_future)
        # x_extra_hist = x_extra_hist+self.sae_hist(x_extra_hist)
        # Concatenate historical time series data with additional historical data
        x_hist = torch.cat([x_hist, x_extra_hist], dim=-1)

        # Transform feature space to time space, apply linear trafo, and convert back
        x_hist_temp = feature_to_time(x_hist)
        x_hist_temp = self.fc_hist(x_hist_temp)
        x_hist = time_to_feature(x_hist_temp)

        # Apply conditional feature mixing to the historical data
        x_hist, _ = self.feature_mixing_hist(x_hist, x_static=x_static)

        # Apply conditional feature mixing to the future data
        x_future, _ = self.feature_mixing_future(x_extra_future, x_static=x_static)

        # Concatenate processed historical and future data
        x = torch.cat([x_hist, x_future], dim=-1)

        # Process the concatenated data through the mixer layers
        for mixing_layer in self.conditional_mixer:
            x = mixing_layer(x, x_static=x_static)

        # Final linear transformation to produce the forecast
        mu = self.fc_out(x)
        sigma = self.fc_sigma(self.fc_presigma(x))

        return mu,sigma,x_extra_future

def cal_saliency(model, all_list,baseline,target_function, steps=50,batch_size=256):
    train_X_hist,train_X_ex_hist,input_seq_list,train_X_static = all_list
    model.train()
    num_samples = input_seq_list.shape[0]
    num_batches = (num_samples + batch_size - 1) // batch_size  # 计算需要多少个批次
    input_seq = torch.Tensor(input_seq_list)
    input_seq.requires_grad=True
    integrated_gradients = torch.zeros_like(input_seq)
    
    for batch_idx in range(num_batches):
    # 计算当前批次的起始和结束索引
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, num_samples)
        
        # 提取当前批次的数据
        input_seq_batch = input_seq[start_idx:end_idx]
        baseline_batch = baseline[start_idx:end_idx]
        train_X_hist_batch = train_X_hist[start_idx:end_idx]
        train_X_ex_hist_batch = train_X_ex_hist[start_idx:end_idx]
        train_X_static_batch = train_X_static[start_idx:end_idx]
        
        # 初始化当前批次的归因值
        integrated_gradients_batch = torch.zeros_like(input_seq_batch)
        
        # 计算步长
        step_size = (input_seq_batch - baseline_batch) / steps
        
        # 对每个积分步进行迭代
        for i in range(steps):
            # 计算当前步的输入
            current_input_data = baseline_batch + (i + 1) * step_size
            # current_input = torch.tensor(current_input_data, requires_grad=True)
            current_input = current_input_data.clone().detach().requires_grad_(True)
            
            # 通过模型获取输出
            # output,sigma,hidden = model(current_input)
            output,sigma,_= model(train_X_hist_batch,train_X_ex_hist_batch,current_input,train_X_static_batch)
            
            # 计算目标值
            target_value = target_function(sigma)
            # target_value = output[:,1]
            
            # 计算当前步的梯度
            grads = torch.autograd.grad(torch.sum(target_value), current_input)[0]
            
            # 更新累计归因值
            integrated_gradients_batch = integrated_gradients_batch +grads**2 
    
        # 将当前批次的归因值存入结果张量中
        integrated_gradients[start_idx:end_idx] = torch.abs(integrated_gradients_batch)
    
    integrated_gradients = integrated_gradients
    
    return integrated_gradients

class TSMixerExt_SAE_attribute(nn.Module):
    """TSMixer model for time series forecasting.

    This model forecasts time series data by integrating historical time series data,
    future known inputs, and static contextual information. It uses a combination of
    conditional feature mixing and mixer layers to process and combine these different
    types of data for effective forecasting.

    Args:
        sequence_length: The length of the input time series sequences.
        prediction_length: The length of the output prediction sequences.
        activation_fn: The name of the activation function to be used.
        num_blocks: The number of mixer blocks in the model.
        dropout_rate: The dropout rate used in the mixer layers.
        input_channels: The number of channels in the historical time series data.
        extra_channels: The number of channels in the extra (future known) inputs.
        hidden_channels: The number of hidden channels used in the mixer layers.
        static_channels: The number of channels in the static feature inputs.
        ff_dim: The inner dimension of the feedforward network in the mixer layers.
        output_channels: The number of output channels for the final output. If None,
                         defaults to the number of input_channels.
        normalize_before: Whether to apply layer normalization before or after mixer layer.
        norm_type: The type of normalization to use. "batch" or "layer".
    """

    def __init__(
        self,
        sequence_length: int,
        prediction_length: int,
        activation_fn: str = "relu",
        num_blocks: int = 2,
        dropout_rate: float = 0.1,
        input_channels: int = 1,
        extra_channels: int = 1,
        hidden_channels: int = 64,
        static_channels: int = 1,
        ff_dim: int = 64,
        output_channels: int = None,
        normalize_before: bool = False,
        norm_type: str = "layer",
    ):
        assert static_channels > 0, "static_channels must be greater than 0"
        super().__init__()

        # Transform activation_fn string to callable function
        if hasattr(F, activation_fn):
            activation_fn = getattr(F, activation_fn)
        else:
            raise ValueError(f"Unknown activation function: {activation_fn}")

        # Transform norm_type to callable
        assert norm_type in {
            "batch",
            "layer",
        }, f"Invalid norm_type: {norm_type}, must be one of batch, layer."
        norm_type = TimeBatchNorm2d if norm_type == "batch" else nn.LayerNorm

        self.fc_hist = nn.Linear(sequence_length, prediction_length)
        self.fc_out = nn.Linear(hidden_channels, output_channels or input_channels)
        self.fc_presigma = nn.Linear(hidden_channels, output_channels or input_channels)
        self.fc_sigma = nn.Softplus()

        self.feature_mixing_hist = ConditionalFeatureMixing(
            sequence_length=prediction_length,
            input_channels=input_channels + extra_channels,
            output_channels=hidden_channels,
            static_channels=static_channels,
            ff_dim=ff_dim,
            activation_fn=activation_fn,
            dropout_rate=dropout_rate,
            normalize_before=normalize_before,
            norm_type=norm_type,
        )
        self.feature_mixing_future = ConditionalFeatureMixing(
            sequence_length=prediction_length,
            input_channels=extra_channels,
            output_channels=hidden_channels,
            static_channels=static_channels,
            ff_dim=ff_dim,
            activation_fn=activation_fn,
            dropout_rate=dropout_rate,
            normalize_before=normalize_before,
            norm_type=norm_type,
        )

        self.conditional_mixer = self._build_mixer(
            num_blocks,
            hidden_channels,
            prediction_length,
            ff_dim=ff_dim,
            static_channels=static_channels,
            activation_fn=activation_fn,
            dropout_rate=dropout_rate,
            normalize_before=normalize_before,
            norm_type=norm_type,
        )
        self.sae_hist = SparseAutoEncoder(input_size=extra_channels, hidden_size=64)  # Assume hidden size is 64
        self.sae_future = SparseAutoEncoder(input_size=extra_channels, hidden_size=64)
        self.x_hist = None
        self.x_extra_hist = None
        self.x_static = None

    @staticmethod
    def _build_mixer(
        num_blocks: int, hidden_channels: int, prediction_length: int, **kwargs
    ):
        """Build the mixer blocks for the model."""
        channels = [2 * hidden_channels] + [hidden_channels] * (num_blocks - 1)

        return nn.ModuleList(
            [
                ConditionalMixerLayer(
                    input_channels=in_ch,
                    output_channels=out_ch,
                    sequence_length=prediction_length,
                    **kwargs,
                )
                for in_ch, out_ch in zip(channels[:-1], channels[1:])
            ]
        )

    def forward(self,x_extra_future: torch.Tensor) -> torch.Tensor:
        """Forward pass for the TSMixer model.

        Processes historical and future data, along with static features, to produce a
        forecast.

        Args:
            x_hist: Historical time series data (batch_size, sequence_length,
                input_channels).
            x_extra_hist: Additional historical data (batch_size, sequence_length,
                extra_channels).
            x_extra_future: Future known data (batch_size, prediction_length,
                extra_channels).
            x_static: Static contextual data (batch_size, static_channels).

        Returns:
            The output tensor representing the forecast (batch_size, prediction_length,
            output_channels).
        """
        x_hist,x_extra_hist,x_static = self.x_hist,self.x_extra_hist,self.x_static
        if x_hist.shape[0]!=x_extra_future.shape[0]:
            x_hist = x_hist.repeat(x_extra_future.shape[0]//x_hist.shape[0], 1, 1)
            x_extra_hist = x_extra_hist.repeat(x_extra_future.shape[0]//x_extra_hist.shape[0], 1, 1)
            x_static = x_static.repeat(x_extra_future.shape[0]//x_static.shape[0], 1)
        else:
            pass
        # x_extra_hist,hidden_hist = self.sae_hist(x_extra_hist)
        x_extra_future = x_extra_future+self.sae_future(x_extra_future)
        # x_extra_hist = x_extra_hist+self.sae_hist(x_extra_hist)
        # Concatenate historical time series data with additional historical data
        x_hist = torch.cat([x_hist, x_extra_hist], dim=-1)

        # Transform feature space to time space, apply linear trafo, and convert back
        x_hist_temp = feature_to_time(x_hist)
        x_hist_temp = self.fc_hist(x_hist_temp)
        x_hist = time_to_feature(x_hist_temp)
        # Apply conditional feature mixing to the historical data
        x_hist, _ = self.feature_mixing_hist(x_hist, x_static=x_static)
        # Apply conditional feature mixing to the future data
        x_future, _ = self.feature_mixing_future(x_extra_future, x_static=x_static)
        # Concatenate processed historical and future data
        x = torch.cat([x_hist, x_future], dim=-1)
        # Process the concatenated data through the mixer layers
        for mixing_layer in self.conditional_mixer:
            x = mixing_layer(x, x_static=x_static)

        # Final linear transformation to produce the forecast
        mu = self.fc_out(x)
        sigma = self.fc_sigma(self.fc_presigma(x))
        return sigma

def cal_captum(train_X_hist,train_X_ex_hist,train_X_ex_fu,train_X_static,model,function,batch_size = 64):
    setup_seed(1)

    # 将数据分割成大小为 batch_size 的批次
    batches1 = torch.chunk(train_X_hist, train_X_hist.size(0) // batch_size + 1, dim=0)
    batches2 = torch.chunk(train_X_ex_hist, train_X_ex_hist.size(0) // batch_size + 1, dim=0)
    batches3 = torch.chunk(train_X_ex_fu, train_X_ex_fu.size(0) // batch_size + 1, dim=0)
    batches4 = torch.chunk(train_X_static, train_X_static.size(0) // batch_size + 1, dim=0)

    # 使用 zip 函数将四个 batches 合并成一个新的 batches
    batches = zip(batches1, batches2, batches3, batches4)

    # 初始化一个空的列表来保存结果
    results = []
    attribution_model = function(model)
    # 对每个批次进行操作
    for batch in batches:
        input1, input2, input3, input4 = batch
        baseline3 = torch.zeros_like(input3).to(device)
        model.x_hist = input1
        model.x_extra_hist = input2
        model.x_static = input4
        attribution, delta = attribution_model.attribute(input3,baseline3, target=0, return_convergence_delta=True)
        attribution_batch = torch.abs(attribution.squeeze(-1)).cpu().detach().numpy()
        results.append(attribution_batch)

    # 将结果拼接起来
    final_result = np.concatenate(results, axis=0)
    return(final_result)

def main(args):
    setup_seed(1)
    name = args.name
    device = args.device
    if not os.path.exists('./result'):
        os.makedirs('./result')
    if not os.path.exists('./result/'+name):
        os.makedirs('./result/'+name)
    train_X_hist = torch.Tensor(np.load('./process_data/'+name+'/train_X_hist.npy')).to(device)
    val_X_hist = torch.Tensor(np.load('./process_data/'+name+'/val_X_hist.npy')).to(device)
    test_X_hist = torch.Tensor(np.load('./process_data/'+name+'/test_X_hist.npy')).to(device)

    train_X_ex_hist = torch.Tensor(np.load('./process_data/'+name+'/train_X_ex_hist.npy')).to(device)
    val_X_ex_hist = torch.Tensor(np.load('./process_data/'+name+'/val_X_ex_hist.npy')).to(device)
    test_X_ex_hist = torch.Tensor(np.load('./process_data/'+name+'/test_X_ex_hist.npy')).to(device)

    train_X_ex_fu = torch.Tensor(np.load('./process_data/'+name+'/train_X_ex_fu.npy')).to(device)
    val_X_ex_fu = torch.Tensor(np.load('./process_data/'+name+'/val_X_ex_fu.npy')).to(device)
    test_X_ex_fu = torch.Tensor(np.load('./process_data/'+name+'/test_X_ex_fu.npy')).to(device)

    train_y = torch.Tensor(np.load('./process_data/'+name+'/train_y.npy')).to(device)
    val_y = torch.Tensor(np.load('./process_data/'+name+'/val_y.npy')).to(device)
    test_y = torch.Tensor(np.load('./process_data/'+name+'/test_y.npy')).to(device)

    train_X_static = torch.Tensor(np.load('./process_data/'+name+'/train_X_static.npy')).to(device)
    val_X_static = torch.Tensor(np.load('./process_data/'+name+'/val_X_static.npy')).to(device)
    test_X_static = torch.Tensor(np.load('./process_data/'+name+'/test_X_static.npy')).to(device)
    model = TSMixerExt_SAE(
    sequence_length=24,
    prediction_length=24,
    input_channels=1,
    extra_channels=test_X_ex_fu.shape[-1],
    hidden_channels=64,
    static_channels=1,
    output_channels=1
    ).to(device)
    model.load_state_dict(torch.load('./model/'+name+'/Tsmixer_SAE.pth'))

    baseline = torch.zeros_like(torch.Tensor(test_X_ex_fu)).to(device)
    saliency = cal_saliency(model, (torch.Tensor(test_X_hist).to(device),torch.Tensor(test_X_ex_hist).to(device),
                                    torch.Tensor(test_X_ex_fu).to(device),torch.Tensor(test_X_static).to(device))
                            ,baseline, mysigma)
    saliency = saliency.cpu().detach().squeeze(-1).numpy()

    saliency_map = np.sum(saliency,1)
    rank = np.sum(saliency_map,0)
    np.save('./result/'+name+'/pinball_rank.npy',rank)

    model = TSMixerExt_SAE_attribute(
    sequence_length=24,
    prediction_length=24,
    input_channels=1,
    extra_channels=test_X_ex_fu.shape[-1],
    hidden_channels=64,
    static_channels=1,
    output_channels=1
    ).to(device)
    model.load_state_dict(torch.load('./model/'+name+'/Tsmixer_SAE.pth'))

    final_result = cal_captum(test_X_hist,test_X_ex_hist,test_X_ex_fu,test_X_static,model,IntegratedGradients)
    saliency_map = np.sum(final_result,1)
    rank = np.sum(saliency_map,0)
    np.save('./result/'+name+'/pinball_IG.npy',rank)

    final_result = cal_captum(test_X_hist,test_X_ex_hist,test_X_ex_fu,test_X_static,model,GradientShap)
    saliency_map = np.sum(final_result,1)
    rank = np.sum(saliency_map,0)
    np.save('./result/'+name+'/pinball_GS.npy',rank)

    final_result = cal_captum(test_X_hist,test_X_ex_hist,test_X_ex_fu,test_X_static,model,DeepLift)
    saliency_map = np.sum(final_result,1)
    rank = np.sum(saliency_map,0)
    np.save('./result/'+name+'/pinball_DL.npy',rank)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n","--name", help="name of the data",default='Spain',choices=['Spain','Panama'])
    parser.add_argument("-d","--device",help="device to run",default='cuda')
    args = parser.parse_args()
    main(args)
    