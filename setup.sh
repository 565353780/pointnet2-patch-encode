cd ..
git clone https://github.com/565353780/mesh-manage.git
git clone https://github.com/565353780/udf-generate.git
git clone https://github.com/565353780/scannet-dataset-manage.git
git clone https://github.com/565353780/scan2cad-dataset-manage.git
git clone https://github.com/565353780/shapenet-dataset-manage.git

cd mesh-manage
./setup.sh

cd ../udf-generate
./setup.sh

cd ../scannet-dataset-manage
./setup.sh

cd ../scan2cad-dataset-manage
./setup.sh

cd ../shapenet-dataset-manage
./setup.sh

pip install open3d numpy pickle tqdm
pip3 install torch torchvision torchaudio \
  --extra-index-url https://download.pytorch.org/whl/cu113

