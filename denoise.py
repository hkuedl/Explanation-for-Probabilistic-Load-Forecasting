import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import *
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error as MSE
from torch.distributions.normal import Normal
import warnings
warnings.filterwarnings("ignore")
from layers import (
    ConditionalFeatureMixing,
    ConditionalMixerLayer,
    TimeBatchNorm2d,
    feature_to_time,
    time_to_feature,
)
from scipy.stats import norm
import pickle
import argparse

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

def predict(model, test_X_hist,test_X_ex_hist,test_X_ex_fu,test_X_static, device, batch_size=256):
    # 定义一个列表来保存预测结果和标准差
    preds = []
    sigmas = []
    
    # 将模型放置在相应的设备上
    model = model.to(device)

    # 确保模型处于评估模式
    model.eval()

    # 循环预测
    for i in range(0, len(test_X_hist), batch_size):
        # 获取批量数据
        batch_X_hist = test_X_hist[i:i+batch_size].to(device)
        batch_X_ex_hist = test_X_ex_hist[i:i+batch_size].to(device)
        batch_X_ex_fu = test_X_ex_fu[i:i+batch_size].to(device)
        batch_X_static = test_X_static[i:i+batch_size].to(device)

        # 预测
        with torch.no_grad():
            pred, sigma, _ = model(batch_X_hist,batch_X_ex_hist,batch_X_ex_fu,batch_X_static)

        # 将预测结果添加到列表
        preds.append(pred.detach().cpu())
        sigmas.append(sigma.detach().cpu())

    # 将所有预测结果拼接在一起
    preds = torch.cat(preds, dim=0).cpu().detach().numpy()
    sigmas = torch.cat(sigmas, dim=0).cpu().detach().numpy()

    return preds, sigmas
def pinball_loss(y_true, y_pred_mean, y_pred_std, tau_list=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]):
    def single_pinball_loss(y_true, y_pred_mean, y_pred_std, tau):
        cdf = norm.cdf(y_true, loc=y_pred_mean, scale=y_pred_std)
        err = y_true - y_pred_mean
        loss = np.where(cdf >= tau, tau * err, (tau - 1) * err)
        return loss.mean()

    total_loss = 0

    # 遍历所有的分位数
    for tau in tau_list:
        total_loss += single_pinball_loss(y_true, y_pred_mean, y_pred_std, tau)

    # 取平均
    total_loss /= len(tau_list)

    return total_loss
def replace_top_elements(testX_real, testX_fake, rank, number):
    # 克隆testX_fake以避免修改原始值
    testX_real,testX_fake = testX_real.cpu().detach().numpy(),testX_fake.cpu().detach().numpy()
    testX_fake_modified = np.copy(testX_fake)

    # 获取前number个最大值的索引
    top_indices = np.argsort(rank)[-number:]
    
    # 将test_fake_modified的对应位置替换为test_real的对应位置
    if number ==0:
        pass
    else:
        for i in top_indices:
            testX_fake_modified[:, :, i] = testX_real[:, :, i]
    
    return torch.Tensor(testX_fake_modified)


def main(args):
    name = args.name
    device = args.device
    use_filter = args.use_filter
    for noise_type in tqdm(['oridinary','random','student']):
        setup_seed(1)
        test_X_hist = torch.Tensor(np.load('./process_data/'+name+'/test_X_hist.npy')).to(device)
        test_X_ex_hist = torch.Tensor(np.load('./process_data/'+name+'/test_X_ex_hist.npy')).to(device)

        test_X_ex_fu = torch.Tensor(np.load('./process_data/'+name+'/test_X_ex_fu.npy')).to(device)

        if noise_type =='oridinary':
            test_X_ex_fu10 = torch.Tensor(np.load('./process_data/'+name+'/test_X_ex_fu_10%.npy')).to(device)
            test_X_ex_fu20 = torch.Tensor(np.load('./process_data/'+name+'/test_X_ex_fu_20%.npy')).to(device)
            test_X_ex_fu30 = torch.Tensor(np.load('./process_data/'+name+'/test_X_ex_fu_30%.npy')).to(device)
            test_X_ex_fu40 = torch.Tensor(np.load('./process_data/'+name+'/test_X_ex_fu_40%.npy')).to(device)
            test_X_ex_fu50 = torch.Tensor(np.load('./process_data/'+name+'/test_X_ex_fu_50%.npy')).to(device)
            test_X_ex_fu60 = torch.Tensor(np.load('./process_data/'+name+'/test_X_ex_fu_60%.npy')).to(device)
            test_X_ex_fu70 = torch.Tensor(np.load('./process_data/'+name+'/test_X_ex_fu_70%.npy')).to(device)
            test_X_ex_fu80 = torch.Tensor(np.load('./process_data/'+name+'/test_X_ex_fu_80%.npy')).to(device)

        elif noise_type == 'random':
            test_X_ex_fu10 = torch.Tensor(np.load('./process_data/'+name+'/test_X_ex_fu_10%_f.npy')).to(device)
            test_X_ex_fu20 = torch.Tensor(np.load('./process_data/'+name+'/test_X_ex_fu_20%_f.npy')).to(device)
            test_X_ex_fu30 = torch.Tensor(np.load('./process_data/'+name+'/test_X_ex_fu_30%_f.npy')).to(device)
            test_X_ex_fu40 = torch.Tensor(np.load('./process_data/'+name+'/test_X_ex_fu_40%_f.npy')).to(device)
            test_X_ex_fu50 = torch.Tensor(np.load('./process_data/'+name+'/test_X_ex_fu_50%_f.npy')).to(device)
            test_X_ex_fu60 = torch.Tensor(np.load('./process_data/'+name+'/test_X_ex_fu_60%_f.npy')).to(device)
            test_X_ex_fu70 = torch.Tensor(np.load('./process_data/'+name+'/test_X_ex_fu_70%_f.npy')).to(device)
            test_X_ex_fu80 = torch.Tensor(np.load('./process_data/'+name+'/test_X_ex_fu_80%_f.npy')).to(device)
        elif noise_type == 'student':
            test_X_ex_fu10 = torch.Tensor(np.load('./process_data/'+name+'/test_X_ex_fu_10%_t.npy')).to(device)
            test_X_ex_fu20 = torch.Tensor(np.load('./process_data/'+name+'/test_X_ex_fu_20%_t.npy')).to(device)
            test_X_ex_fu30 = torch.Tensor(np.load('./process_data/'+name+'/test_X_ex_fu_30%_t.npy')).to(device)
            test_X_ex_fu40 = torch.Tensor(np.load('./process_data/'+name+'/test_X_ex_fu_40%_t.npy')).to(device)
            test_X_ex_fu50 = torch.Tensor(np.load('./process_data/'+name+'/test_X_ex_fu_50%_t.npy')).to(device)
            test_X_ex_fu60 = torch.Tensor(np.load('./process_data/'+name+'/test_X_ex_fu_60%_t.npy')).to(device)
            test_X_ex_fu70 = torch.Tensor(np.load('./process_data/'+name+'/test_X_ex_fu_70%_t.npy')).to(device)
            test_X_ex_fu80 = torch.Tensor(np.load('./process_data/'+name+'/test_X_ex_fu_80%_t.npy')).to(device)

        test_y = torch.Tensor(np.load('./process_data/'+name+'/test_y.npy')).to(device)
        test_X_static = torch.Tensor(np.load('./process_data/'+name+'/test_X_static.npy')).to(device)

        rank = np.load('./result/'+name+'/pinball_rank.npy')
        IGrank = np.load('./result/'+name+'/pinball_IG.npy')
        GSrank = np.load('./result/'+name+'/pinball_GS.npy')
        DLrank = np.load('./result/'+name+'/pinball_DL.npy')
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
        iter_list = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20] if name =='Spain' else [0,1,2,3,4,5,6,7,8]

        setup_seed(1)
        my_list_list,IG_list_list,GS_list_list,DL_list_list,random_list_list = [],[],[],[],[]
        for noise_input in [test_X_ex_fu10,test_X_ex_fu20,test_X_ex_fu30,test_X_ex_fu40,test_X_ex_fu50,test_X_ex_fu60,test_X_ex_fu70,test_X_ex_fu80]:
            if use_filter:
                fs = 50      # 采样频率
                order = 2
                if noise_input is test_X_ex_fu10:
                    cutoff = 15
                else:
                    if name =='Spain':
                        cutoff = 8
                    else:
                        cutoff = 10
                test_X_ex_fu = lowpass_filter_denoise_tensor(noise_input.cpu(), cutoff, fs, order)
            my_list = []
            for i in iter_list:
                input_X = replace_top_elements(test_X_ex_fu,noise_input,rank,i).to(device)
                preds, sigmas = predict(model, test_X_hist,test_X_ex_hist,input_X,test_X_static, device)
                my_list.append(pinball_loss(test_y.cpu().detach().numpy(), preds, sigmas))
            my_list = np.array(my_list)
            my_list_list.append(my_list)
            IG_list = []
            for i in iter_list:
                input_X = replace_top_elements(test_X_ex_fu,noise_input,IGrank,i).to(device)
                preds, sigmas = predict(model, test_X_hist,test_X_ex_hist,input_X,test_X_static, device)
                IG_list.append(pinball_loss(test_y.cpu().detach().numpy(), preds, sigmas))
            IG_list = np.array(IG_list)
            IG_list_list.append(IG_list)

            GS_list = []
            for i in iter_list:
                input_X = replace_top_elements(test_X_ex_fu,noise_input,GSrank,i).to(device)
                preds, sigmas = predict(model, test_X_hist,test_X_ex_hist,input_X,test_X_static, device)
                GS_list.append(pinball_loss(test_y.cpu().detach().numpy(), preds, sigmas))
            GS_list = np.array(GS_list)
            GS_list_list.append(GS_list)

            DL_list = []
            for i in iter_list:
                input_X = replace_top_elements(test_X_ex_fu,noise_input,DLrank,i).to(device)
                preds, sigmas = predict(model, test_X_hist,test_X_ex_hist,input_X,test_X_static, device)
                DL_list.append(pinball_loss(test_y.cpu().detach().numpy(), preds, sigmas))
            DL_list = np.array(DL_list)
            DL_list_list.append(DL_list)

            setup_seed(1)
            fake_rank = np.random.rand(test_X_ex_fu.shape[-1])
            random_list = []
            for i in iter_list:
                input_X = replace_top_elements(test_X_ex_fu,noise_input,fake_rank,i).to(device)
                preds, sigmas = predict(model, test_X_hist,test_X_ex_hist,input_X,test_X_static, device)
                random_list.append(pinball_loss(test_y.cpu().detach().numpy(), preds, sigmas))
            random_list = np.array(random_list)
            random_list_list.append(random_list)

        area_our_list,area_random_list,area_IG_list,area_GS_list,area_DL_list = [],[],[],[],[]
        for i in range(8):
            area_ours = np.trapz(my_list_list[i], dx=1)
            area_our_list.append(area_ours)
            area_random = np.trapz(random_list_list[i], dx=1)
            area_random_list.append(area_random)
            area_IG = np.trapz(IG_list_list[i], dx=1)
            area_IG_list.append(area_IG)
            area_GS = np.trapz(GS_list_list[i], dx=1)
            area_GS_list.append(area_GS)
            area_DL = np.trapz(DL_list_list[i], dx=1)
            area_DL_list.append(area_DL)
        result = pd.DataFrame([area_random_list,area_IG_list,area_GS_list,area_DL_list,area_our_list])
        result = result.round(4)
        result.columns = ['0.1', '0.2','0.3','0.4','0.5','0.6','0.7','0.8']
        result.index = ['Random', 'IG', 'GS', 'DL', 'Ours']
        print(result)
        if use_filter:
            result.to_json('./result/'+name+'/'+noise_type+'.json', orient='records', lines=True)
        else:
            result.to_json('./result/'+name+'/'+noise_type+'_low_filter.json', orient='records', lines=True)
        
    
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n","--name", help="name of the data",default='Spain',choices=['Spain','Panama'])
    parser.add_argument("-d","--device",help="device to run",default='cuda')
    parser.add_argument("-u","--use_filter",help="whether to use low-pass filter",default=False)
    args = parser.parse_args()
    main(args)