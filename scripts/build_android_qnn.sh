#!/bin/bash
mkdir ../build-arm-qnn
cd ../build-arm-qnn || exit

export ANDROID_NDK_ROOT="/home/cc/workspace/android-ndk-r28"

cmake .. \
-DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK_ROOT/build/cmake/android.toolchain.cmake \
-DCMAKE_BUILD_TYPE=Release \
-DANDROID_ABI="arm64-v8a" \
-DANDROID_STL=c++_static \
-DANDROID_NATIVE_API_LEVEL=android-28  \
-DNATIVE_LIBRARY_OUTPUT=. -DNATIVE_INCLUDE_OUTPUT=. $1 $2 $3 \
-DQNN=ON \
-DDEBUG=OFF \
-DTEST=OFF \
-DQUANT=ON \
-DQNN_VALIDATE_NODE=OFF \
-DMLLM_BUILD_XNNPACK_BACKEND=OFF

make -j$(nproc)
