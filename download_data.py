
# The download_data.py file would automatically download the data for you to train the network
data_dir = './data'

import assistance

assistance.download_extract('mnist', data_dir)
assistance.download_extract('celeba', data_dir)