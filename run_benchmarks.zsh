#!/bin/zsh


# ssh -p 1983 -t sniper@155.207.33.164 "cd /home/sniper/tensorflow_env; /./home/sniper/.local/bin/pipenv run"
model_name="googlenet"
params="--model=$model_name --trace_file=/home/sniper/tf_cnn_benchmarks/scripts/tf_cnn_benchmarks/$model_name"
scp -P 1983 -r ./scripts sniper@155.207.33.164://home/sniper/tf_cnn_benchmarks/
# echo ';deactivate' | ssh -p 1983 -t sniper@155.207.33.164 "source /home/sniper/tensorflow_env/bin/activate; python tf_cnn_benchmarks/scripts/tf_cnn_benchmarks/tf_cnn_benchmarks.py $params"

# scp -P 1983 -r ./tools sniper@155.207.33.164://home/sniper/tf_cnn_benchmarks/
# echo ';deactivate' | ssh -p 1983 -t sniper@155.207.33.164 "source /home/sniper/tensorflow_env/bin/activate; python tf_cnn_benchmarks/tools/run_all_CNN_benchmarks.py"

# scp -P 1983 -r ./my_tests sniper@155.207.33.164://home/sniper/tf_cnn_benchmarks/
# echo ';deactivate' | ssh -p 1983 -t sniper@155.207.33.164 "source /home/sniper/tensorflow_env/bin/activate; python tf_cnn_benchmarks/my_tests/reportLmdbError.py"