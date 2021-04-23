import os 
import zipfile
import requests

def download_extract(url):
   """
   Download the movielens dataset.
    
   """
   BASE_DIR = './data/'
   
   os.makedirs(f'{BASE_DIR}', exist_ok=True)
   fname = os.path.join(BASE_DIR, url.split('/')[-1])
   
   print(f'Downloading {fname} from {url}...')
   r = requests.get(url, stream=True, verify=False)
   with open(fname, 'wb') as f:
       f.write(r.content)
   
   base_dir = os.path.dirname(fname)
   data_dir, ext = os.path.splitext(fname)
   
   if ext == '.zip':
       fp = zipfile.ZipFile(fname, 'r')
   fp.extractall(base_dir)
   print('Extract is Done')
   
