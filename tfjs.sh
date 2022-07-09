tensorflowjs_converter \
    --input_format=tf_saved_model \
    --output_node_names='MobilenetV3s/Output' \
    --saved_model_tags=serve \
    /home/wgzhong/pywork/age-gender-recognise/outputs/2022-07-02/01-59-14/save_model \
    ./tfjs/demo/serverweb_model