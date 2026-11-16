set -euo pipefail

pythonBin=.venv/bin/python

echo "== clone llama.cpp"
git clone https://github.com/ggml-org/llama.cpp build/llama.cpp
git -C build/llama.cpp checkout b12375
"$pythonBin" -m pip install -r build/llama.cpp/requirements.txt

echo "== cmake build"
cmake -S build/llama.cpp -B build/llama.cpp/build-host -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF
cmake --build build/llama.cpp/build-host -j"$(nproc)" --target llama-quantize llama-imatrix llama-perplexity llama-server

echo "== ARM build Cortex-A8"
cmake -S build/llama.cpp -B build/llama.cpp/build-armv7 -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_SYSTEM_NAME=Linux -DCMAKE_SYSTEM_PROCESSOR=arm \
  -DCMAKE_C_COMPILER=arm-linux-gnueabihf-gcc -DCMAKE_CXX_COMPILER=arm-linux-gnueabihf-g++ \
  -DCMAKE_C_FLAGS="$armFlags" -DCMAKE_CXX_FLAGS="$armFlags" \
  -DCMAKE_EXE_LINKER_FLAGS="-static" \
  -DGGML_NATIVE=OFF -DGGML_LLAMAFILE=OFF -DGGML_OPENMP=OFF -DBUILD_SHARED_LIBS=OFF -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF
cmake --build build/llama.cpp/build-armv7 -j"$(nproc)" --target llama-server

cp build/llama.cpp/build-armv7/bin/llama-server build/out/llama-server
arm-linux-gnueabihf-strip build/out/llama-server

echo "== convert and quantize train/merged"
"$pythonBin" build/llama.cpp/convert_hf_to_gguf.py train/merged --outtype f16 --outfile build/out/beagle-f16.gguf
build/llama.cpp/build-host/bin/llama-imatrix -m build/out/beagle-f16.gguf -f data/calib.txt \
  -o build/out/imatrix-beagle.dat -c 512
build/llama.cpp/build-host/bin/llama-quantize --imatrix build/out/imatrix-beagle.dat \
  build/out/beagle-f16.gguf build/out/beagle-q4_k_m.gguf Q4_K_M

echo "== convert and quantize train/base"
"$pythonBin" build/llama.cpp/convert_hf_to_gguf.py train/base --outtype f16 --outfile build/out/base-f16.gguf
build/llama.cpp/build-host/bin/llama-imatrix -m build/out/base-f16.gguf -f data/calib.txt \
  -o build/out/imatrix-base.dat -c 512
build/llama.cpp/build-host/bin/llama-quantize --imatrix build/out/imatrix-base.dat \
  build/out/base-f16.gguf build/out/base-q4_k_m.gguf Q4_K_M

echo "-- beagle f16"
build/llama.cpp/build-host/bin/llama-perplexity -m build/out/beagle-f16.gguf -f data/eval_text.txt -c 512 --chunks 100 2>&1 \
  | tee -a build/out/perplexity.log | grep "Final estimate"
echo "-- beagle Q4_K_M"
build/llama.cpp/build-host/bin/llama-perplexity -m build/out/beagle-q4_k_m.gguf -f data/eval_text.txt -c 512 --chunks 100 2>&1 \
  | tee -a build/out/perplexity.log | grep "Final estimate"
 
echo "Done!"