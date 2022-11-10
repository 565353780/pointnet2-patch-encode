cd ..
git clone git@github.com:565353780/mesh-manage.git
git clone git@github.com:565353780/udf-generate.git
git clone git@github.com:565353780/scannet-dataset-manage.git
git clone git@github.com:565353780/scan2cad-dataset-manage.git
git clone git@github.com:565353780/shapenet-dataset-manage.git

cd mesh-manage
./dev_setup.sh

cd ../udf-generate
./dev_setup.sh

cd ../scannet-dataset-manage
./dev_setup.sh

cd ../scan2cad-dataset-manage
./dev_setup.sh

cd ../shapenet-dataset-manage
./dev_setup.sh

pip install open3d numpy pickle tqdm
pip3 install torch torchvision torchaudio \
  --extra-index-url https://download.pytorch.org/whl/cu113

