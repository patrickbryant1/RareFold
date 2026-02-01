#Install conda env
conda env create -f environment.yml

wait
conda activate rarefold
pip install -q --no-warn-conflicts numpy==1.26.4
pip install -q --no-warn-conflicts 'jax[cuda12_pip]'==0.4.35 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
conda deactivate

## Get network parameters for RareFold (a few minutes)
#RareFold + EvoBindRare params
wget https://zenodo.org/records/18420884/files/params50000.npy
mkdir data/params
mv params50000.npy data/params/

wait
## Get Uniclust30 (10-20 minutes depending on bandwidth)
# 25 Gb download, 87 Gb extracted
wget http://wwwuser.gwdg.de/~compbiol/uniclust/2018_08/uniclust30_2018_08_hhsuite.tar.gz --no-check-certificate
mv uniclust30_2018_08_hhsuite.tar.gz data
cd data
tar -zxvf uniclust30_2018_08_hhsuite.tar.gz
cd ..

wait
## Install HHblits (a few minutes)
git clone https://github.com/soedinglab/hh-suite.git
mkdir -p hh-suite/build && cd hh-suite/build
cmake -DCMAKE_INSTALL_PREFIX=. ..
make -j 4 && make install
cd ../..

wait
