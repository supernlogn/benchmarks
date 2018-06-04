
model_list=("alexnet" "densenet" "googlenet" "inception" "vgg11" "vgg16" "vgg19" "lenet" "overfeat" "trivial" "inception3" "inception4" "official_resnet18_v2" "official_resnet34_v2" "official_resnet50_v2" "official_resnet101_v2" "official_resnet152_v2" "official_resnet200_v2" "official_resnet18" "official_resnet34" "official_resnet50" "official_resnet101" "official_resnet152" "official_resnet200" "resnet50" "resnet50_v2" "resnet101" "resnet101_v2" "resnet152" "resnet152_v2" "nasnet" "mobilenet" "squeezenet")


for model_name in $model_list; do
    python tf_cnn_benchmarks/tf_cnn_benchmarks.py --model=$model_name --trace_file=/home/sniper/tf_cnn_benchmarks/scripts/tf_cnn_benchmarks/$model_name
    mv "/home/sniper/tf_cnn_benchmarks/trace_data.ctf" "/home/sniper/tf_cnn_benchmarks/trace_data_$modelname.ctf"
    mv "logfile.txt" "logfile_$model_name.txt"
done