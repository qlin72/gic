DATA_DIR="/data/dataset_45_new"

for i in $(seq 0 8); do
    for j in $(seq $((i + 1)) 9); do
        pair_index=$(printf "%d%d" "$i" "$j")
        echo "Preparing data for pair: $pair_index"
        python prepare_pacnerf_data.py --data_folder="$DATA_DIR/$pair_index"
    done
done