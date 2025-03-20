git config --global user.email "642741043@qq.com"
git config --global user.name "Wonderful-Me"

# init mllm repo
# git clone git@github.com:Wonderful-Me/mllm.git
# cd mllm
# git submodule update --init --recursive
# cd ..

# gcc-13 update
sudo add-apt-repository ppa:ubuntu-toolchain-r/test -y
sudo apt update
sudo apt install gcc-13 g++-13 -y
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-13 100
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-13 100

# cmake update
sudo apt update
sudo apt remove cmake  # remove old version first
sudo apt install -y software-properties-common
sudo apt-key adv --fetch-keys "https://apt.kitware.com/keys/kitware-archive-latest.asc"
sudo apt-add-repository -y "deb https://apt.kitware.com/ubuntu/ $(lsb_release -cs) main"
sudo apt update
sudo apt install -y cmake

# package installation
sudo apt install unzip -y
sudo apt install python -y
sudo apt install libncurses5 -y 

# QNN SDK
wget https://dl.google.com/android/repository/android-ndk-r28-linux.zip
unzip android-ndk-r28-linux.zip
rm android-ndk-r28-linux.zip
export ANDROID_NDK_ROOT="/home/cc/workspace/android-ndk-r28"

# # Install QPM3 from https://qpm.qualcomm.com/#/main/tools/details/QPM3
# qpm-cli --login yx102@rice.edu

# # QNN SDK
# # Path: ./mllm/src/backends/qnn/sdk/bin/envsetup.sh
# qpm-cli --license-activate qualcomm_neural_processing_sdk
# qpm-cli --extract qualcomm_neural_processing_sdk
# mv /opt/qcom/aistack/qairt/2.31.0.250130 ./mllm/src/backends/qnn/sdk
# source ./mllm/src/backends/qnn/sdk/bin/envsetup.sh
# source /home/cc/workspace/mllm/src/backends/qnn/sdk/bin/envsetup.sh

# # Hexagon SDK
# # Path: /local/mnt/workspace/Qualcomm/Hexagon_SDK/5.5.3.0
# qpm-cli --license-activate HexagonSDK5.x
# qpm-cli --install HexagonSDK5.x -y
# source /local/mnt/workspace/Qualcomm/Hexagon_SDK/5.5.3.0/setup_sdk_env.source