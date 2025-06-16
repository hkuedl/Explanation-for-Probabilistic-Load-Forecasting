# Explanation-for-Probabilistic-Load-Forecasting

## Add data

Please download the data in the [Cloud Drive](https://connecthkuhk-my.sharepoint.com/personal/u3009646_connect_hku_hk/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fu3009646%5Fconnect%5Fhku%5Fhk%2FDocuments%2Fdata%2Fdata%2Fexplanation&ga=1) and put it in the root path.

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

Note that the name should be in ['Spain', 'Panama'].
