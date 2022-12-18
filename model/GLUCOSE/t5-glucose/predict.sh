MODEL_DIR=large
INPUT_FILE=dim7-dev-glucose
OUTPUT_FILE=dim7-dev-glucose-target

t5_mesh_transformer \
  --model_dir="${MODEL_DIR}" \
  --gin_file="${MODEL_DIR}/operative_config.gin" \
  --gin_file="infer.gin" \
  --gin_file="beam_search.gin" \
  --gin_param="input_filename = 'data/${INPUT_FILE}.txt'"\
  --gin_param="output_filename = 'outputs/${OUTPUT_FILE}.txt'"\
  --gin_param="utils.run.mesh_shape = 'model:1,batch:1'" \
  --gin_param="utils.run.mesh_devices = ['gpu:2']" \
  --gin_param="infer_checkpoint_step = 'all'"