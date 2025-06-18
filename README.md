# Explanation-for-Probabilistic-Load-Forecasting
Codes for Paper "A Unified Explanation Framework for Probabilistic Load Forecasting via Feature Uncertainty Propagation".

Authors: Zhixian Wang, Linxiao Yang, Chenxi Wang, Liang Sun, and Yi Wang.

## Create Environment

Please use the following command to create the environment.

~~~
conda create --name exp python=3.8
conda activate exp
pip install -r requirements.txt
~~~

## Add data

Please download the data in the [Cloud Drive](https://connecthkuhk-my.sharepoint.com/:f:/g/personal/u3009646_connect_hku_hk/EoQK6vwviVtGlDlwzE7P7FoBNAvVKwyGZ-pFxJu0qMrmKg?e=s1G89D) and put it in the root path.

## Calculate the importance

Please use the following command to calculate the importance.

~~~

python cal_rank.py -n 'Spain'

~~~

Note that the name should be in ['Spain', 'Panama'].

## Denoising process

Please use the following command to get the denoising result.

~~~

python denoise.py -n 'Spain'

~~~

Note that the name should be in ['Spain', 'Panama']. Then you can get the result in the result folder.
