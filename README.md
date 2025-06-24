# Explanation-for-Probabilistic-Load-Forecasting

Codes for Paper "A Unified Explanation Framework for Probabilistic Load Forecasting via Feature Uncertainty Propagation".

Authors: Zhixian Wang, Linxiao Yang, Chenxi Wang, Liang Sun, Fernando Porté-Agel, Yi Wang.

## Down load the data

Please download the data in the cloud drive [Cloud Drive](https://connecthkuhk-my.sharepoint.com/:f:/g/personal/u3009646_connect_hku_hk/Ei9s3K5jNWVCjfa_9y6OgKkBaG7AqIUM9xDQML2aB7NUJg?e=fxDso7) and place them in the root path.

## Calculate the importance

Please use the following command to calculate the feature importance.

~~~
python cal_rank.py -n Spain
~~~

Please note that the dataset name should be in ['Spain','Panama'].


## Calculate the importance

Please use the following command to get the denoise result.

~~~
python denoise.py -n Spain -u False
~~~

Please note that the dataset name should be in ['Spain','Panama'] and use -u True to apply to low-pass filter.
