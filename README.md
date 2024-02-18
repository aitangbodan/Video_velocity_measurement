# Video velocity measurement

The real-time video analysis algorithm module based on tensorrt dynamic batch inference accesses the upper layer service at the model inference output end. 
Business involved:***Video velocity measurement***


[TOC]

### main module description

**model**

The algorithm serves the underlying algorithm implementation : tracking algorithm, optical flow algorithm. 

**mutual**

Interactive interface with the back-end : mainly related to grpc communication and shared memory read and write.

**parallel**

Parallel Task Management Module : Tasks that need to be processed in parallel can be placed here and currently include tracking tasks.

**application**

Business application module : including the underlying algorithm scheduling, business configuration, business table, etc.



### configuration

#### **Algorithm configuration**

Modify the underlying algorithm configuration, see ***“model/config/config.yaml”***

De:
  model_path: './model/engine_file/model.432FP32.engine'
  input_name: 'inputs'
  output_name: 'outputs'
  cates: [ 'bg','water' ]
  half: False



#### **Business configuration**

pass in a json object. The configuration file format is detailed in ***"./application/config/water_flow_velocity_config.json*"**

```
[
  {
    "camera": "001",
    "rois": 2,
    "rois_shape": [[432, 131], [432, 131]],
    "left_top": [[0, 0], [0, 0]],
    "distance": 12.3,
    "change": []
  },
  {
    "camera": "002",
    "rois": 1,
    "rois_shape": [[432, 131]],
    "left_top": [[0, 0]],
    "distance": 11.0,
    "change": [0]
  }]
```

### Run

#### **Local testing**

Enter the project root directory, business testing local test. 

```
python module_test/manager_test.py
```

#### **back-end joint debugging**

 	Start the grpc server and run python grpc/smserver.py

​	













