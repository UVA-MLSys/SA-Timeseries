import os, shutil, pyunpack, requests

# General functions for data downloading & aggregation.
def download_from_url(url, output_path):
  """Downloads a file froma url."""

  print('\nPulling data from {} to {}'.format(url, output_path))
  with requests.get(url, allow_redirects=True, stream=True) as response:
    response.raise_for_status()

    with open(output_path, 'wb') as outfile:
      for chunk in response.iter_content(): 
        # If you have chunk encoded response uncomment if
        # and set chunk_size parameter to None.
        #if chunk: 
        outfile.write(chunk)

  print('done.\n')


def recreate_folder(path):
  """Deletes and recreates folder."""

  shutil.rmtree(path)
  os.makedirs(path)


def unzip(zip_path, output_file, data_folder):
  """Unzips files and checks successful completion."""

  print('Unzipping file: {}'.format(zip_path))
  pyunpack.Archive(zip_path).extractall(data_folder)

  # Checks if unzip was successful
  if not os.path.exists(output_file):
    raise ValueError(
        'Error in unzipping process! {} not found.'.format(output_file))


def cleanup(directory:str, csv_path:str):
  """AI is creating summary for cleanup

  Args:
      directory (str): [description]
      csv_path (str): [description]
  """
  print('\nRemoving unnecessary files.')

  for item in os.listdir(directory):
    item_path = os.path.join(os.path.join(directory, item))
    if item_path == csv_path: continue

    if os.path.isdir(item_path):
        shutil.rmtree(item_path)
    else:
      os.remove(item_path)

    print(f'Removed {item}')
  
  print('Cleaning completed.\n')
    

def download_and_unzip(url, zip_path, csv_path, data_folder):
  """Downloads and unzips an online csv file.

  Args:
    url: Web address
    zip_path: Path to download zip file
    csv_path: Expected path to csv file
    data_folder: Folder in which data is stored.
  """
  download_from_url(url, zip_path)
  unzip(zip_path, csv_path, data_folder)

  print('Done.')