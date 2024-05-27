import os
import tarfile
import argparse
import requests
from urllib.request import urlretrieve  # 正确导入 urlretrieve
# 连接不上dropbox：
#
# 打开Ping服务器响应速度检测网址：http://ping.chinaz.com/
# 输入dropbox.com， 回车，获取各个服务器响应ip的速度，
# 修改host文件：打开路径C/Widnows/System32/drivers/etc/下面的hosts文件，在文件末尾输入如下信息，并保存修改。
#
# # access to dropbox start
# 162.125.248.18 www.dropbox.com
# 162.125.248.18 dl-web.dropbox.com
# 162.125.248.18 dl.dropboxusercontent.com
# # dropbox end


SOURCES = {
    'mnist': 'https://www.dropbox.com/s/rzurpt5gzb14a1q/pretrained_mnist.tar',
    'anime': 'https://www.dropbox.com/s/9aveavgbluvjeu6/pretrained_anime.tar',
    'biggan': 'https://www.dropbox.com/s/zte4oein08ajsij/pretrained_biggan.tar',
    'proggan': 'https://www.dropbox.com/s/707xjn1rla8nwqc/pretrained_proggan.tar',
    'stylegan2': 'https://www.dropbox.com/s/c3aaq7i6soxmpzu/pretrained_stylegan2_ffhq.tar',
}


def download(source, destination):
    tmp_tar = os.path.join(destination, '.tmp.tar')
    # urllib has troubles with dropbox
    os.system(f'wget --no-check-certificate {source} -O {tmp_tar}')
    tar_file = tarfile.open(tmp_tar, mode='r')
    tar_file.extractall(destination)

    # 使用urllib
    # Ensure the destination directory exists
    # os.makedirs(destination, exist_ok=True)
    #
    # # Use urllib to download the file
    # print(f'Downloading from {source} to {tmp_tar}')
    # try:
    #     urlretrieve(source, tmp_tar)
    # except Exception as e:
    #     raise FileNotFoundError(f"Failed to download file from {source}: {e}")
    #
    # # Check if the file was downloaded
    # if not os.path.exists(tmp_tar):
    #     raise FileNotFoundError(f"Failed to download file from {source}")
    #
    # # Extract the tar file
    # with tarfile.open(tmp_tar, mode='r') as tar_file:
    #     tar_file.extractall(destination)

    # proxies = {
    #     "http": os.environ.get("http_proxy"),
    #     "https": os.environ.get("https_proxy"),
    # }
    #
    #
    # # 使用requests
    # os.makedirs(destination, exist_ok=True)
    #
    # # Use requests to download the file
    # print(f'Downloading from {source} to {tmp_tar}')
    # try:
    #     with requests.get(source, stream=True, verify=False,proxies=proxies) as r:
    #         r.raise_for_status()
    #         with open(tmp_tar, 'wb') as f:
    #             for chunk in r.iter_content(chunk_size=8192):
    #                 if chunk:  # filter out keep-alive new chunks
    #                     f.write(chunk)
    # except Exception as e:
    #     raise FileNotFoundError(f"Failed to download file from {source}: {e}")
    #
    # # Check if the file was downloaded
    # if not os.path.exists(tmp_tar):
    #     raise FileNotFoundError(f"Failed to download file from {source}")
    #
    # # Check the file size
    # file_size = os.path.getsize(tmp_tar)
    # print(f'Downloaded file size: {file_size} bytes')
    #
    # # Extract the tar file
    # try:
    #     with tarfile.open(tmp_tar, mode='r') as tar_file:
    #         tar_file.extractall(destination)
    # except tarfile.ReadError as e:
    #     raise tarfile.ReadError(f"Failed to open tar file: {e}")


    os.remove(tmp_tar)


def main():
    parser = argparse.ArgumentParser(description='Pretrained models loader')
    parser.add_argument('--models', nargs='+', type=str,
                        choices=list(SOURCES.keys()) + ['all'], default=['all'])
    parser.add_argument('--out', type=str, help='root out dir')

    args = parser.parse_args()

    if args.out is None:
        args.out = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'models')

    models = args.models
    if 'all' in models:
        models = list(SOURCES.keys())

    for model in set(models):
        source = SOURCES[model]
        print(f'downloading {model}\nfrom {source}')
        download(source, args.out)


if __name__ == '__main__':
    main()
