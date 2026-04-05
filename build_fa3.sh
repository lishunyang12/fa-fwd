set -e
source ./set_compile_thread.sh
# set_compile_params
####################### Build FA3 #######################
# Set environment variables to control FlashAttention build options

export FLASH_ATTENTION_DISABLE_BACKWARD=TRUE
export FLASH_ATTENTION_DISABLE_FP16=TRUE
export FLASH_ATTENTION_DISABLE_LOCAL=TRUE
export FLASH_ATTENTION_FORCE_BUILD=TRUE
export FLASH_ATTENTION_DISABLE_PAGEDKV=TRUE


# clone submodule if not exists
if [ ! -d "flash-attention" ]; then
    git submodule update --init --recursive
fi

mkdir -p build

cp hopper_setup_py.patch flash-attention/
cp two_level_accum.patch flash-attention/

cd flash-attention

git checkout .

git apply --check hopper_setup_py.patch
git apply hopper_setup_py.patch

git apply --check two_level_accum.patch
git apply two_level_accum.patch

cd hopper

mv flash_attn_interface.py fa3_fwd_interface.py

echo "__version__ = \"0.0.2\"" > __init__.py

sed -i 's/flash_attn_3/fa3_fwd/g' fa3_fwd_interface.py flash_api_stable.cpp flash_api.cpp

NVCC_THREADS=$NVCC_THREADS MAX_JOBS=$MAX_JOBS \
    python setup.py bdist_wheel --verbose

cp dist/*.whl ../../build/
