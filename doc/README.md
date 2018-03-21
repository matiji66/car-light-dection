### start from this page  
 ```https://github.com/tensorflow/models/tree/master/research/object_detection```
 
#### 0. prepare image and label image to xml

#### 1. xml_to_csv.py

#### 2. split_labes.py 

#### 3. draw_boxes.py display bounding box


#### 4. generate_tfrecord.py
 
##### No Need From tensorflow/models/
  
  Create train data:
  python generate_tfrecord.py --csv_input=data/train_labels.csv  --output_path=data/train.record

  # Create test data:
  python generate_tfrecord.py --csv_input=data/test_labels.csv  --output_path=data/test.record
  

  
#### 5. confi pipline  

5.1 config checkpoint 
```
While optional, it is highly recommended that users utilize other object detection checkpoints. Training an object detector from scratch can take days. To speed up the training process, it is recommended that users re-use the feature extractor parameters from a pre-existing object classification or detection checkpoint. train_config provides two fields to specify pre-existing checkpoints: fine_tune_checkpoint and from_detection_checkpoint. fine_tune_checkpoint should provide a path to the pre-existing checkpoint (ie:"/usr/home/username/checkpoint/model.ckpt-#####"). 
# from_detection_checkpoint is a boolean value. If false, it assumes the checkpoint was from an object classification checkpoint. Note that starting from a detection checkpoint will usually result in a faster training job than a classification checkpoint.
```

fine_tune_checkpoint: "/usr/home/username/tmp/model.ckpt-#####"
from_detection_checkpoint: true

5.2 config train_input_reader
```
train_input_reader: {
  tf_record_input_reader {
    input_path: "../data/train.record"
  }
  label_map_path: "../data/light_label_map.pbtxt"
}

eval_config: {
  num_examples: 40
} 
```


5.3 config eval_input_reader 

```
eval_input_reader: {
  tf_record_input_reader {
    input_path: "../data/test.record"
  }
  label_map_path: "../data/light_label_map.pbtxt"
  shuffle: false
  num_readers: 1
}
``` 

#### 6. train data 
  
  #### running locally  
  https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/running_locally.md
  
  
  ```
  python train.py --train_dir='/home/shz/TF-OD-Test/train' --pipeline_config_path='/home/shz/TF-OD-Test/models/ssd_mobilenet/ssd_mobilenet_v1_pascal.config'
  ```

